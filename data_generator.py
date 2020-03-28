#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import argparse
from functools import partial
from typing import List, Dict, Tuple

import numpy as np

from config import EVAL_FRAMERATE_SCALE
from data.dataset import fetch_dataset, fetch_dataset_pair
from data.types import DriveDataFrame
from game_environment import DaggerGeneratorEnvironment, OfflineGeneratorEnvironment
from parameter import Parameter
from util.common import add_carla_module, get_logger
from util.road_option import fetch_index_from_road_option

logger = get_logger(__name__)
add_carla_module()


def align_indices_from_dicts(
        image_frame_dict: Dict[int, np.ndarray],
        drive_frame_dict: Dict[int, DriveDataFrame],
        search_range: int) -> Tuple[bool, List[int], List[int]]:
    image_frame_keys = sorted(image_frame_dict.keys())
    drive_frame_keys = sorted(drive_frame_dict.keys())
    min_frame_key = min(image_frame_keys[0], drive_frame_keys[0])
    max_frame_key = max(image_frame_keys[-1], drive_frame_keys[-1])
    reference_frame_range = list(range(min_frame_key, max_frame_key + 1, EVAL_FRAMERATE_SCALE))
    rel_indices = list(filter(lambda x: x != 0, range(-search_range, search_range + 1)))
    image_indices, drive_indices = [], []
    for reference_frame_key in reference_frame_range:
        if reference_frame_key in image_frame_dict:
            image_indices.append(reference_frame_key)
        else:
            found = False
            for rel in rel_indices:
                if reference_frame_key + rel in image_frame_dict:
                    image_indices.append(reference_frame_key + rel)
                    found = True
                    break
            if not found:
                logger.error('could not find a proper neighboring image at {}'.format(reference_frame_key))
                return False, [], []
        if reference_frame_key in drive_frame_dict:
            drive_indices.append(reference_frame_key)
        else:
            found = False
            for rel in rel_indices:
                if reference_frame_key + rel in drive_frame_dict:
                    drive_indices.append(reference_frame_key + rel)
                    found = True
                    break
            if not found:
                logger.error('could not find a proper neighboring drive frame at {}'.format(reference_frame_key))
                return False, [], []
    assert image_indices
    assert len(image_indices) == len(drive_indices)
    return True, image_indices, drive_indices


def load_dagger_generator(args):
    port = args.port
    ports = args.ports
    assert ports
    assert port in ports
    assert args.dataset_name
    assert args.eval_dataset_name
    port_index = ports.index(port)

    eval_param = Parameter()
    eval_param.exp_index = args.exp_index
    eval_param.exp_name = args.exp_name
    eval_param.load()
    eval_param.batch_size = 1
    eval_param.dataset_data_names = [args.dataset_name]
    eval_param.eval_data_name = args.eval_dataset_name
    eval_param.max_data_length = -1

    if eval_param.split_train:
        train_dataset, valid_dataset = fetch_dataset_pair(eval_param)
    else:
        train_dataset = fetch_dataset(eval_param)

    num_data = len(train_dataset)
    index_func = partial(fetch_index_from_road_option, low_level=eval_param.use_low_level_segment)
    data_list = []
    for i in range(num_data):
        # road_option, data_frame = train_dataset.get_trajectory_data(i)
        # images = data_frame.images
        # drives = data_frame.drives
        # fixme: this may not work if train_dataset is a HighLevelDataset? Cuz it does not even have this method!
        road_option, images, drives = train_dataset.get_trajectory_data(i)
        data_list.append({
            'road_option': road_option,
            'action_index': index_func(road_option),
            'src_transform': drives[0].state.transform,
            'dst_location': drives[-1].state.transform.location,
            'length': len(images)
        })

    def chunker_list(seq, size):
        return (seq[i::size] for i in range(size))

    index_data_lists = list(chunker_list(list(enumerate(data_list)), len(ports)))
    index_data_list = index_data_lists[port_index]
    return DaggerGeneratorEnvironment(args, eval_param, index_data_list)


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(description='Data generator')
    argparser.add_argument('data_type', type=str)
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='10.158.54.63', help='host server IP (default: 10.158.54.63')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port (default: 2000)')
    argparser.add_argument('--ports', type=int, nargs='*', default=[])
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='200x88', help='window resolution (default: 200x88)')
    argparser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    argparser.add_argument('--map', metavar='TOWN', default=None, help='start a new episode at the given TOWN')
    argparser.add_argument('--speed', metavar='S', default=20, type=int, help='Maximum speed in PID controller')
    argparser.add_argument('--no-rendering', action='store_true', help='switch off server rendering')
    argparser.add_argument('--show-triggers', action='store_true', help='show trigger boxes of traffic signs')
    argparser.add_argument('--show-connections', action='store_true', help='show waypoint connections')
    argparser.add_argument('--show-spawn-points', action='store_true', help='show recommended spawn points')
    argparser.add_argument('--safe', action='store_true', help='avoid spawning vehicles prone to accidents')
    argparser.add_argument('--show-game', action='store_true', help='show game display')
    argparser.add_argument('--eval-timestamp', type=int)
    argparser.add_argument('--exp-name', type=str)
    argparser.add_argument('--exp-step', type=int, default=None)
    argparser.add_argument('--exp-cmd', type=int, default=0)
    argparser.add_argument('--exp-index', type=int, default=32)
    argparser.add_argument('--random-seed', type=int, default=0)
    argparser.add_argument('--position-index', type=int, default=0)
    argparser.add_argument('--dataset-name', type=str)
    argparser.add_argument('--eval-dataset-name', type=str)
    argparser.add_argument('--camera-keywords', type=str, nargs='*', default=['left', 'center', 'right'])

    args = argparser.parse_args()
    args.description = argparser.description
    args.width, args.height = [int(x) for x in args.res.split('x')]

    if args.data_type == 'dagger':
        gen_env = load_dagger_generator(args=args)
    elif args.data_type == 'offline':
        gen_env = OfflineGeneratorEnvironment(args=args)
    else:
        raise TypeError('invalid data type {}'.format(args.data_type))
    gen_env.run()


if __name__ == '__main__':
    main()
