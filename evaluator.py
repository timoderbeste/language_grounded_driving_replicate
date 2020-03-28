#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import argparse
import json
import random
from collections import defaultdict
from functools import partial
from math import sqrt
from operator import itemgetter
from pathlib import Path
from typing import Tuple, List, Any

from custom_carla.agents.navigation.local_planner import RoadOption
from data.types import DriveDataFrame
from evaluation_environment import AllOfflineEvaluationEnvironment, AllOnlineEvaluationEnvironment
from util.common import add_carla_module, get_logger
from util.directory import mkdir_if_not_exists

add_carla_module()
logger = get_logger(__name__)
import carla

import numpy as np
import pygame
import torch

from config import IMAGE_WIDTH, IMAGE_HEIGHT
from data.dataset import load_index_from_word, generate_templated_sentence_dict, HighLevelDataset
from util.road_option import fetch_road_option_from_str, fetch_onehot_vector_dim, fetch_onehot_vector_from_index, \
    fetch_num_sentence_commands, fetch_onehot_vector_from_sentence_command, fetch_onehot_index_from_high_level_str
from util.image import tensor_from_numpy_image
from model import init_hidden_states
from parameter import Parameter
from checkpoint_base import CheckpointBase


def canonicalize(src: str):
    return str(src).replace('\\', '/').replace('//', '/')


def onehot_from_index(cmd: int, use_low_level_segment: bool) -> torch.Tensor:
    onehot_dim = fetch_onehot_vector_dim(use_low_level_segment)
    return fetch_onehot_vector_from_index(cmd, use_low_level_segment).view(1, onehot_dim)


def _tensor_from_numpy_image(image: np.ndarray) -> torch.Tensor:
    c = image.shape[2]
    return tensor_from_numpy_image(image, False).view(1, 1, c, IMAGE_HEIGHT, IMAGE_WIDTH)


class LowLevelEvaluator(CheckpointBase):
    def __init__(self, param: Parameter, cmd: int):
        CheckpointBase.__init__(self, param)
        self.cmd = cmd
        self.param = param
        self.step_elapsed = 0
        self.use_low_level_segment = param.use_low_level_segment
        self.onehot_dim = fetch_onehot_vector_dim(param.use_low_level_segment)
        self.onehot_func = partial(onehot_from_index, use_low_level_segment=param.use_low_level_segment)
        self.encoder_hidden, self.decoder_hidden, self.images = None, None, []
        self.initialize()

    def initialize(self):
        self.encoder_hidden, self.decoder_hidden = init_hidden_states(self.param)
        self.images = []

    def run_step(self, image: np.ndarray) -> Any:
        batch = self._prepare_batch(image)
        output = self._run_step(batch)
        self.step_elapsed += 1
        return output

    @property
    def onehot_vector(self):
        return fetch_onehot_vector_from_index(self.cmd, self.use_low_level_segment).view(1, self.onehot_dim)

    def _prepare_batch(self, image: np.ndarray, custom_action_index: int = -1):
        # self.initialize()
        self.images.append(_tensor_from_numpy_image(image))
        self.images = self.images[-10:]
        data_dict = {
            'onehot': self.onehot_vector,
            'action_index': [self.cmd if custom_action_index < 0 else custom_action_index],
            'images': torch.cat(self.images, dim=1)
        }
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device=self.param.device_type)
        return data_dict

    def _run_step(self, data):
        self.model.eval()
        # todo: what type of model is used here? find it out in the paper!
        model_output = self.model.forward(data, self.encoder_hidden, self.decoder_hidden)
        output = model_output['output'][0][-1]
        if output.size(-1) == 2:
            control = carla.VehicleControl()
            control.throttle = output[0].item()
            control.steer = output[1].item()
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control
        else:
            return output.item()


class HighLevelEvaluator(CheckpointBase):
    def __init__(self, param: Parameter, cmd: str):
        CheckpointBase.__init__(self, param)
        self.cmd = cmd
        self.param = param
        self.step_elapsed = 0
        self.onehot_dim = fetch_num_sentence_commands()
        self.onehot_func = fetch_onehot_vector_from_sentence_command
        self.index_func = fetch_onehot_index_from_high_level_str
        self.sentence = param.eval_keyword.lower()
        self.index_from_word = load_index_from_word()
        self.encoder_hidden, self.decoder_hidden, self.images = None, None, []
        self.initialize()

    def fetch_word_index(self, word: str):
        if word in self.index_from_word:
            return self.index_from_word[word]
        else:
            return 1  # assigned for 'unknown'

    def initialize(self):
        self.encoder_hidden, self.decoder_hidden = init_hidden_states(self.param)
        self.images = []

    def run_step(self, image: np.ndarray, sentence: str) -> torch.Tensor:
        batch = self._prepare_batch(image, sentence)
        action = self._run_step(batch)
        return action

    @property
    def onehot_vector(self):
        return self.onehot_func(self.cmd).view(1, self.onehot_dim)

    def _prepare_batch(self, image: np.ndarray, sentence: str):
        word_indices = [self.fetch_word_index(w) for w in sentence.lower().split(' ')]
        length = torch.tensor([len(word_indices)], dtype=torch.long)
        logger.info((length.shape, sentence))
        word_indices = torch.tensor(word_indices, dtype=torch.long)
        self.images.append(_tensor_from_numpy_image(image))
        self.images = self.images[-10:]
        data_dict = {
            'sentence': sentence,
            'word_indices': word_indices,
            'length': length,
            'images': torch.cat(self.images, dim=1)
        }
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device=self.param.device_type)
        return data_dict

    def _run_step(self, data):
        self.model.eval()
        model_output = self.model(data, self.encoder_hidden, self.decoder_hidden)
        agent_action = model_output['output'][0]
        self.step_elapsed += 1
        return agent_action


class SingleEvaluator(CheckpointBase):
    def __init__(self, param: Parameter, cmd: int):
        CheckpointBase.__init__(self, param)
        self.cmd = cmd
        self.param = param
        self.step_elapsed = 0
        self.use_low_level_segment = param.use_low_level_segment
        self.index_from_word = load_index_from_word()
        self.onehot_dim = fetch_onehot_vector_dim(param.use_low_level_segment)
        self.onehot_func = partial(onehot_from_index, use_low_level_segment=param.use_low_level_segment)
        self.encoder_hidden, self.decoder_hidden, self.images = None, None, []
        self.initialize()

    def fetch_word_index(self, word: str):
        if word in self.index_from_word:
            return self.index_from_word[word]
        else:
            return 1  # assigned for 'unknown'

    def initialize(self):
        self.encoder_hidden, self.decoder_hidden = init_hidden_states(self.param)
        self.images = []

    def run_step(self, image: np.ndarray, sentence: str) -> Any:
        batch = self._prepare_batch(image, sentence)
        output = self._run_step(batch)
        self.step_elapsed += 1
        return output

    @property
    def onehot_vector(self):
        return fetch_onehot_vector_from_index(self.cmd, self.use_low_level_segment).view(1, self.onehot_dim)

    def _prepare_batch(self, image: np.ndarray, sentence: str, custom_action_index: int = -1):
        # self.initialize()
        word_indices = [self.fetch_word_index(w) for w in sentence.lower().split(' ')]
        length = torch.tensor([len(word_indices)], dtype=torch.long)
        word_indices = torch.tensor(word_indices, dtype=torch.long)
        self.images.append(_tensor_from_numpy_image(image))
        self.images = self.images[-10:]
        data_dict = {
            'sentence': sentence,
            'word_indices': word_indices,
            'length': length,
            'images': torch.cat(self.images, dim=1)
        }
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device=self.param.device_type)
        return data_dict

    def _run_step(self, data):
        self.model.eval()
        model_output = self.model.forward(data, self.encoder_hidden, self.decoder_hidden)
        output = model_output['output'][0][-1]
        if output.size(-1) == 2:
            control = carla.VehicleControl()
            control.throttle = output[0].item()
            control.steer = output[1].item()
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control
        else:
            return output.item()


class EvaluationMetaInfo:
    def __init__(self, timestamp: int, road_option: RoadOption, frame_range: Tuple[int, int]):
        self.timestamp = timestamp
        self.road_option = road_option
        self.frame_range = frame_range


def str_from_road_option(road_option: RoadOption) -> str:
    return road_option.name.lower()


def road_option_from_str(road_option_str: str) -> RoadOption:
    return fetch_road_option_from_str(road_option_str)


def _prepare_evaluation_param(param: Parameter) -> Parameter:
    assert param.eval_data_name
    assert param.eval_info_name
    assert param.eval_keyword
    if param.model_level == 'low':
        param.eval_keyword = fetch_road_option_from_str(param.eval_keyword.upper())
    elif param.model_level == 'high':
        param.eval_keyword = param.eval_keyword.lower()
    else:
        logger.info(param.model_level)
        raise TypeError('invalid eval_keyword was given {}'.format(param.eval_keyword))
    param.max_data_length = -1
    param.shuffle = False
    param.batch_size = 1
    param.dataset_data_names = [param.eval_data_name]
    param.dataset_info_names = [param.eval_info_name]
    if param.model_level == 'low':
        param.use_multi_cam = False
        param.use_sequence = False
        param.has_clusters = False
    return param


def fetch_unique_data_from_high_level_dataset(dataset: HighLevelDataset, eval_keyword: str) -> \
        List[Tuple[List[DriveDataFrame], str]]:
    keywords = dataset.get_extended_keywords()
    indices = list(map(itemgetter(0), filter(lambda x: x[1].lower() == eval_keyword, enumerate(keywords))))

    def position_from_drive_data_frame(drive_frame: DriveDataFrame):
        location = drive_frame.state.transform.location
        return location.x, location.y

    positions = []
    for i in indices:
        drive_list = dataset.get_mid_drive_data(i)
        xs, ys = zip(*list(map(position_from_drive_data_frame, drive_list)))
        positions.append((list(xs), list(ys)))

    def compute_list_dist(l1, l2) -> float:
        return sqrt(sum([(v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2 for v1, v2 in zip(l1, l2)]))

    dist_threshold = 10.0
    edge_dict = defaultdict(list)
    for i, v1 in enumerate(positions):
        ivl = sorted([(j, compute_list_dist(v1, positions[j])) for j in range(i + 1, len(positions))],
                     key=itemgetter(1))
        edge_dict[i] = list(map(itemgetter(0), filter(lambda x: x[1] < dist_threshold, ivl)))

    visited = [False for _ in range(len(positions))]
    unique_indices = []
    for i in range(len(positions)):
        if visited[i]:
            continue
        visited[i] = True
        for n in edge_dict[i]:
            visited[n] = True
        unique_indices.append(i)

    indices = [indices[i] for i in unique_indices]

    data_list = []
    for i in indices:
        keyword, sentence, data_frame = dataset.get_trajectory_data_from_sequence_index(i)
        data_list.append((data_frame.drives, sentence))
    return data_list


def load_evaluation_dataset(param: Parameter) -> Tuple[List[List[DriveDataFrame]], List[str]]:
    param = _prepare_evaluation_param(param)
    data_root = Path.cwd() / '.carla/dataset/evaluation'
    if int(param.dataset_data_names[0][-1]) == 1:
        data_root = data_root / 'town1'
    else:
        data_root = data_root / 'town2'
    if not data_root.exists():
        raise FileNotFoundError('could not find {}'.format(data_root))

    data_path = data_root / '{}.json'.format(param.eval_keyword)
    with open(str(data_path), 'r') as file:
        eval_dict = json.load(file)

    drives = [[DriveDataFrame.load_from_str(d) for d in dl] for dl in eval_dict['drives']]
    sentences = eval_dict['sentences']
    return list(drives), list(sentences)


def load_param_and_evaluator(eval_keyword: str, args, model_type: str):
    param = Parameter()
    low_level = model_type in ['control', 'stop']
    if model_type == 'control':
        exp_index = args.control_model_index
        exp_name = args.control_model_name
        exp_step = args.control_model_step
    elif model_type == 'stop':
        exp_index = args.stop_model_index
        exp_name = args.stop_model_name
        exp_step = args.stop_model_step
    elif model_type == 'high':
        exp_index = args.high_level_index
        exp_name = args.high_level_name
        exp_step = args.high_level_step
    elif model_type == 'single':
        exp_index = args.single_model_index
        exp_name = args.single_model_name
        exp_step = args.single_model_step
    else:
        raise TypeError('invalid model type {}'.format(model_type))
    param.exp_name = exp_name
    param.exp_index = exp_index
    param.load()
    param.batch_size = 1
    param.eval_keyword = eval_keyword
    param.eval_data_name = args.eval_data_name
    param.eval_info_name = args.eval_info_name

    logger.info('model type: {}'.format(param.model_type))
    cls = LowLevelEvaluator if low_level else (SingleEvaluator if model_type == 'single' else HighLevelEvaluator)
    logger.info((model_type, cls, param.model_level, param.encoder_type))
    eval_arg = args.exp_cmd if low_level else eval_keyword
    evaluator = cls(param, eval_arg)
    evaluator.load(step=exp_step)
    return param, evaluator


def listen_keyboard() -> str:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 'q'
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return 'q'
            elif event.key == pygame.K_l:
                return 'l'
            elif event.key == pygame.K_r:
                return 'r'
            elif event.key == pygame.K_s:
                return 's'
            elif event.key == pygame.K_f:
                return 'f'
            elif event.key == pygame.K_u:
                return 'u'
            elif event.key == pygame.K_i:
                return 'i'
            elif event.key == pygame.K_o:
                return 'o'
            elif event.key == pygame.K_j:
                return 'j'
            elif event.key == pygame.K_k:
                return 'k'
            elif event.key == pygame.K_m:
                return 'm'
            elif event.key == pygame.K_COMMA:
                return ','
            elif event.key == pygame.K_PERIOD:
                return '.'
            elif event.key == pygame.K_1:
                return '1'
            elif event.key == pygame.K_2:
                return '2'
            elif event.key == pygame.K_3:
                return '3'
            elif event.key == pygame.K_4:
                return '4'
            elif event.key == pygame.K_5:
                return '5'
    return ''


__keyword_from_input__ = {
    'j': 'left',
    'k': 'straight',
    'l': 'right',
    'u': 'left,left',
    'i': 'left,straight',
    'o': 'left,right',
    'm': 'right,left',
    ',': 'right,straight',
    '.': 'right,right',
    '1': 'straight,straight',
    '2': 'firstleft',
    '3': 'firstright',
    '4': 'secondleft',
    '5': 'secondright'
}
__input_from_keyword__ = {v: k for k, v in __keyword_from_input__.items()}

__sentence_library_dict__ = generate_templated_sentence_dict()


def get_random_sentence_from_keyword(keyword: str) -> str:
    def replace_word(word: str):
        if word.startswith('extrastraight'):
            return 'straight'
        elif word == 'extraleft':
            return 'left'
        elif word == 'extraright':
            return 'right'
        else:
            return word

    words = list(map(replace_word, keyword.split(',')))
    keyword = ','.join(words)
    if keyword not in __sentence_library_dict__:
        raise KeyError('invalid keyword was given {}'.format(keyword))
    sentence_group = __sentence_library_dict__[keyword]
    sentences = random.choice(sentence_group)
    sentence = random.choice(sentences)
    return sentence


def fetch_ip_address():
    from subprocess import run
    from re import findall
    from subprocess import PIPE
    raw_lines = run(['ifconfig'], stdout=PIPE).stdout.decode()
    candidates = findall('inet addr:([\d]+.[\d]+.[\d]+.[\d]+)', raw_lines)

    def filter_out(cand: str):
        if cand == '192.168.0.1':
            return False
        if cand.startswith('127') or cand.startswith('172'):
            return False
        return True

    candidates = list(filter(filter_out, candidates))
    if candidates:
        return candidates[0]
    else:
        return '172.0.0.1'


class ExperimentArgument:
    def __init__(self, eval_name: str, info_dict: dict):
        exp_keys = ['port', 'keywords', 'data']
        for key in exp_keys:
            if key not in info_dict:
                raise KeyError('essential key was not found {}'.format(key))

        self.eval_type: str = 'offline'
        self.model_level: str = 'both'
        self.verbose: bool = False
        self.host: str = fetch_ip_address()
        self.port: int = info_dict['port']
        self.res: str = '200x88'
        self.width, self.height = [int(x) for x in self.res.split('x')]
        self.filter: str = 'vehicle.*'
        self.map: str = None
        self.speed: int = 20
        self.no_rendering: bool = False
        self.safe: bool = False
        self.show_game: bool = False
        self.eval_name: str = eval_name
        self.eval_keywords: List[str] = info_dict['keywords']
        self.exp_cmd: int = 0
        self.random_seed: int = 0
        self.position_index: int = 0
        self.eval_data_name: str = info_dict['data']['name']
        self.eval_info_name: str = '{}-v{}'.format(self.eval_data_name, info_dict['data']['version'])
        self.camera_keywords: List[str] = ['center']

        model_keys = ['control', 'stop', 'high', 'single']
        model_suffix = ['model', 'model', 'level', 'model']
        items = ['index', 'name', 'step']
        default_values = [None, '', None]
        for key, suffix in zip(model_keys, model_suffix):
            if key in info_dict:
                for item in items:
                    setattr(ExperimentArgument, '{}_{}_{}'.format(key, suffix, item), info_dict[key][item])
            else:
                for item, value in zip(items, default_values):
                    setattr(ExperimentArgument, '{}_{}_{}'.format(key, suffix, item), value)


def launch_experiment_from_json(exp_name: str, online: bool):
    conf_dir = Path.cwd() / '.carla/settings/experiments'
    conf_path = conf_dir / '{}.json'.format(exp_name)
    if not conf_path.exists():
        raise FileNotFoundError('configuration file does not exist {}'.format(conf_path))

    with open(str(conf_path), 'r') as file:
        data = json.load(file)

    def prepare_model(info_dict: dict):
        index, name, step = info_dict['index'], info_dict['name'], info_dict['step']
        rel_checkpoint_dir = '.carla/checkpoints/exp{}/{}'.format(index, name)
        rel_param_dir = '.carla/params/exp{}'.format(index)
        checkpoint_pth_name = 'step{:06d}.pth'.format(step)
        checkpoint_json_name = 'step{:06d}.json'.format(step)
        param_name = '{}.json'.format(name)
        model_dir = Path.cwd() / rel_checkpoint_dir
        param_dir = Path.cwd() / rel_param_dir
        if not model_dir.exists():
            mkdir_if_not_exists(model_dir)
        if not param_dir.exists():
            mkdir_if_not_exists(param_dir)
        checkpoint_model_path = Path.cwd() / '{}/{}'.format(rel_checkpoint_dir, checkpoint_pth_name)
        checkpoint_json_path = Path.cwd() / '{}/{}'.format(rel_checkpoint_dir, checkpoint_json_name)
        param_path = Path.cwd() / '{}/{}'.format(rel_param_dir, param_name)

        error_messages = []
        if not checkpoint_model_path.exists() or not checkpoint_json_path.exists() or not param_path.exists():
            servers = ['dgx:/raid/rohjunha', 'grta:/home/rohjunha']
            from subprocess import run
            for server in servers:
                try:
                    run(['scp', '{}/{}/{}'.format(server, rel_checkpoint_dir, checkpoint_pth_name),
                         checkpoint_model_path])
                    run(['scp', '{}/{}/{}'.format(server, rel_checkpoint_dir, checkpoint_json_name),
                         checkpoint_json_path])
                    run(['scp', '{}/{}/{}'.format(server, rel_param_dir, param_name), param_path])
                except:
                    error_messages.append('file not found in {}'.format(server))
                finally:
                    pass

        if not checkpoint_model_path.exists() or not checkpoint_json_path.exists() or not param_path.exists():
            logger.error(error_messages)
            raise FileNotFoundError('failed to fetch files from other servers')

    model_keys = ['control', 'stop', 'high', 'single']
    for key in model_keys:
        if key in data:
            prepare_model(data[key])

    args = ExperimentArgument(exp_name, data)
    cls = AllOnlineEvaluationEnvironment if online else AllOfflineEvaluationEnvironment
    for keyword in args.eval_keywords:
        eval_env = cls(eval_keyword=keyword, args=args)
        if not eval_env.run():
            break


def main():
    argparser = argparse.ArgumentParser(description='Evaluation of trained models')
    argparser.add_argument('exp_name', type=str)
    argparser.add_argument('--online', action='store_true')
    args = argparser.parse_args()
    launch_experiment_from_json(args.exp_name, args.online)


if __name__ == '__main__':
    main()
