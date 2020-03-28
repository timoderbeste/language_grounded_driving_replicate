import json
import re
from collections import OrderedDict
from pathlib import Path

import torch

from model import fetch_agent
from parameter import Parameter
# from trainer import logger
from util.common import set_random_seed, get_logger
from util.directory import fetch_checkpoint_dir, fetch_checkpoint_path, fetch_checkpoint_meta_path


logger = get_logger(__name__)


class CheckpointSchedulerMixin:
    def __init__(self):
        self.scheduler_steps = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        self.max_scheduled_step = self.scheduler_steps[-1]
        self.scheduler_step_size = 2500

    def save(self, step: int):
        raise NotImplementedError

    def save_scheduled(self, step: int):
        scheduled = step <= self.max_scheduled_step and step in self.scheduler_steps
        regular = step > self.max_scheduled_step and step % self.scheduler_step_size == 0
        if scheduled or regular:
            self.save(step)
            logger.info('checkpoint was saved at step {}'.format(step))


class EvaluateSchedulerMixin:
    def __init__(self):
        self.eval_scheduler_steps = [5000, 10000]
        self.eval_max_scheduled_step = self.eval_scheduler_steps[-1]
        self.eval_scheduler_step_size = 10000

    def evaluate(self, step: int):
        raise NotImplementedError

    def eval_scheduled(self, step: int):
        scheduled = step <= self.eval_max_scheduled_step and step in self.eval_scheduler_steps
        regular = step > self.eval_max_scheduled_step and step % self.eval_scheduler_step_size == 0
        if scheduled or regular:
            self.evaluate(step)
            logger.info('evaluation was requested at step {}'.format(step))


class CheckpointBase(CheckpointSchedulerMixin, EvaluateSchedulerMixin):
    def __init__(self, param: Parameter):
        self.modes = ['train', 'valid'] if param.split_train else ['train']
        self.epochs, self.steps = dict(), dict()
        for mode in self.modes:
            self.epochs[mode] = 0
            self.steps[mode] = 0

        CheckpointSchedulerMixin.__init__(self)
        EvaluateSchedulerMixin.__init__(self)

        self.exp_index = param.exp_index
        self.exp_name = param.exp_name
        self.shuffle = param.shuffle
        self.device_type = param.device_type
        self.model = fetch_agent(param)

        set_random_seed(param.random_seed)

    @property
    def step(self):
        return self.steps['train']

    @property
    def checkpoint_dir(self):
        return fetch_checkpoint_dir(self.exp_index, self.exp_name)

    def epoch_step_str(self, keyword: str):
        return 'epoch {:05d}, step {:06d}'.format(self.epochs[keyword], self.steps[keyword])

    def model_path(self, step: int):
        return fetch_checkpoint_path(self.exp_index, self.exp_name, step)

    def meta_path(self, step: int):
        return fetch_checkpoint_meta_path(self.exp_index, self.exp_name, step)

    def save_model(self, step: int):
        torch.save(self.model.state_dict(), str(self.model_path(step)))

    def save_meta(self, step: int):
        meta = dict()
        for mode in self.modes:
            meta['{}_epoch'.format(mode)] = self.epochs[mode]
            meta['{}_step'.format(mode)] = self.steps[mode]
        with open(str(self.meta_path(step)), 'w') as file:
            json.dump(meta, file, indent=2)

    def save(self, step: int):
        self.save_model(step)
        self.save_meta(step)

    def load_meta(self, step=None):
        if step is not None and step > 0:
            meta_path = self.meta_path(step)
        else:
            meta_path = find_latest_file(self.checkpoint_dir, '.json')
        if meta_path is None or not meta_path.exists():
            logger.error('could not find the meta path {}'.format(meta_path))
            return

        with open(str(meta_path), 'r') as file:
            data = json.load(file)
        logger.info('found meta data {}'.format(data))
        for mode in self.modes:
            self.epochs[mode] = data['{}_epoch'.format(mode)]
            self.steps[mode] = data['{}_step'.format(mode)]

    def load_model(self, step=None):
        if step is not None and step > 0:
            model_path = self.model_path(step)
        else:
            model_path = find_latest_file(self.checkpoint_dir, '.pth')
        if model_path is None or not model_path.exists():
            logger.error('could not find the model path {}'.format(model_path))
            return

        state_dict = torch.load(str(model_path), map_location=self.device_type)
        new_state_dict = OrderedDict()
        logger.info('checkpoint loaded from {}'.format(model_path))
        for key in state_dict.keys():
            if key.startswith('decoder.stop'):
                continue
            elif key.endswith('stop_linear.weight') or key.endswith('stop_linear.bias'):
                continue
            else:
                new_state_dict[key] = state_dict[key]
        self.model.load_state_dict(new_state_dict)

    def load(self, step=None):
        self.load_meta(step)
        self.load_model(step)


def find_latest_file(root_dir: Path, suffix: str) -> Path:
    def fetch_step_from_path(path: Path) -> int:
        result = list(map(lambda x: int(x), re.findall('step([0-9]+)', str(path))))
        return result[0] if result else -1
    path_step_list = [(path, fetch_step_from_path(path)) for path in root_dir.glob('*{}'.format(suffix))]
    path_step_list = list(sorted(path_step_list, key=lambda x: x[1]))
    return path_step_list[-1][0] if path_step_list else None