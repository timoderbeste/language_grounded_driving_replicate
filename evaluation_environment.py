import json
import random
import shutil
from operator import attrgetter
from pathlib import Path
from time import sleep
from typing import Dict

import cv2
import numpy as np
import pygame
import torch

from config import EVAL_FRAMERATE_SCALE, DATASET_FRAMERATE
from data.types import DriveDataFrame, LengthComputer
from game_environment import GameEnvironment, FrameCounter, set_world_asynchronous, set_world_synchronous, should_quit
from evaluator import load_param_and_evaluator, load_evaluation_dataset, logger, get_random_sentence_from_keyword, \
    listen_keyboard, __keyword_from_input__
from util.directory import EvaluationDirectory, mkdir_if_not_exists
from util.image import video_from_files
from util.road_option import fetch_high_level_command_from_index


class EvaluationEnvironmentBase(GameEnvironment, EvaluationDirectory):
    def __init__(self, args, model_type: str):
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')
        self.eval_param, self.evaluator = load_param_and_evaluator(args=args, model_type=model_type)
        self.eval_transforms = self.world.get_map().get_spawn_points()
        EvaluationDirectory.__init__(self, *self.eval_info)

    @property
    def eval_info(self):
        return self.eval_param.exp_index, self.eval_param.exp_name, self.evaluator.step, 'online'


class AllOfflineEvaluationEnvironment(GameEnvironment, EvaluationDirectory):
    def __init__(self, eval_keyword: str, args):
        self.eval_keyword = eval_keyword
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')
        self.eval_name = args.eval_name

        # load params and evaluators
        self.control_param, self.control_evaluator = \
            load_param_and_evaluator(eval_keyword=eval_keyword, args=args, model_type='control')
        self.stop_param, self.stop_evaluator = \
            load_param_and_evaluator(eval_keyword=eval_keyword, args=args, model_type='stop')
        self.high_level_param, self.high_level_evaluator = \
            load_param_and_evaluator(eval_keyword=eval_keyword, args=args, model_type='high')

        # set image type
        self.image_type = self.high_level_param.image_type
        if 'd' in self.image_type:
            from model import DeepLabModel, prepare_deeplab_model
            self.deeplab_model: DeepLabModel = prepare_deeplab_model()

        self.final_images = []
        self.eval_dataset, self.eval_sentences = load_evaluation_dataset(self.high_level_param)
        self.eval_transforms = list(map(lambda x: x[0].state.transform, self.eval_dataset))
        self.high_level_sentences = self.eval_sentences
        logger.info('fetched {} sentences from {}'.format(
            len(self.high_level_sentences), self.high_level_param.eval_keyword.lower()))
        self.softmax = torch.nn.Softmax(dim=1)
        EvaluationDirectory.__init__(self, *self.eval_info)
        self.high_level_data_dict = dict()

    @property
    def eval_info(self):
        return self.control_param.exp_index, self.eval_name, \
               self.control_evaluator.step, self.eval_keyword

    @property
    def segment_image(self):
        return np.reshape(((self.agent.segment_frame[:, :, 2] == 7).astype(dtype=np.uint8) * 255), (88, 200, 1))

    @property
    def custom_segment_image(self):
        return np.reshape(self.deeplab_model.run(self.agent.image_frame), (88, 200, 1))

    @property
    def final_image(self):
        if self.image_type == 's':
            return self.segment_image
        elif self.image_type == 'd':
            return self.custom_segment_image
        elif self.image_type == 'bgr':
            return self.agent.image_frame
        elif self.image_type == 'bgrs':
            return np.concatenate((self.agent.image_frame, self.segment_image), axis=-1)
        elif self.image_type == 'bgrd':
            return np.concatenate((self.agent.image_frame, self.custom_segment_image), axis=-1)
        else:
            raise TypeError('invalid image type {}'.format(self.image_type))

    def export_evaluation_data(self, t: int, curr_eval_data: dict) -> bool:
        with open(str(self.state_path(t)), 'w') as file:
            json.dump(curr_eval_data, file, indent=2)

        data_frames = [DriveDataFrame.load_from_str(s) for s in curr_eval_data['data_frames']]
        controls = list(map(attrgetter('control'), data_frames))
        stops, sub_goals = zip(*curr_eval_data['stop_frames'])
        texts = ['th{:+4.2f} st{:+4.2f} {:4s}:{:+4.2f}'.format(c.throttle, c.steer, g[:4], s)
                 for c, s, g in zip(controls, stops, sub_goals)]
        text_dict = {i: t for i, t in zip(range(*curr_eval_data['frame_range']), texts)}
        src_image_files = [self.agent.image_path(f) for f in range(*curr_eval_data['frame_range'])]
        src_image_files = list(filter(lambda x: x.exists(), src_image_files))
        if self.image_type in ['s', 'd']:
            final_image_files = [self.segment_dir / '{:08d}.png'.format(i) for i in range(len(self.final_images))]
            for p, s in zip(final_image_files, self.final_images):
                cv2.imwrite(str(p), s)
            video_from_files(final_image_files, self.video_dir / 'segment{:02d}.mp4'.format(t),
                             texts=[], framerate=EVAL_FRAMERATE_SCALE * DATASET_FRAMERATE, revert=False)
        image_frames = set([int(s.stem[:-1]) for s in src_image_files])
        drive_frames = set(text_dict.keys())
        common_frames = sorted(list(image_frames.intersection(drive_frames)))
        src_image_files = [self.agent.image_path(f) for f in common_frames]
        dst_image_files = [self.image_dir / p.name for p in src_image_files]
        [shutil.copy(str(s), str(d)) for s, d in zip(src_image_files, dst_image_files)]
        text_list = [text_dict[f] for f in common_frames]
        video_from_files(src_image_files, self.video_path(t),
                         texts=text_list, framerate=EVAL_FRAMERATE_SCALE * DATASET_FRAMERATE, revert=True)
        return self.state_path(t).exists()

    def run_single_trajectory(self, t: int, transform: carla.Transform) -> Dict[str, bool]:
        status = {
            'exited': False,  # has to finish the entire loop
            'finished': False,  # this procedure has been finished successfully
            'saved': False,  # successfully saved the evaluation data
            'collided': False,  # the agent has collided
            'restart': False,  # this has to be restarted
            'stopped': True  # low-level controller returns stop
        }
        self.agent.reset()
        self.agent.move_vehicle(transform)
        self.control_evaluator.initialize()
        self.stop_evaluator.initialize()
        self.high_level_evaluator.initialize()
        self.high_level_data_dict[t] = []
        self.final_images = []
        sentence = random.choice(self.high_level_sentences)
        logger.info('moved the vehicle to the position {}'.format(t))

        count = 0
        frame = None
        clock = pygame.time.Clock() if self.show_image else FrameCounter()

        set_world_asynchronous(self.world)
        sleep(0.5)
        set_world_synchronous(self.world)

        agent_len, expert_len = LengthComputer(), LengthComputer()
        for l in self.eval_dataset[t]:
            expert_len(l.state.transform.location)
        criterion_len = 2.5 * expert_len.length  # 0.9 * expert_len.length
        max_iter = 10.0 * len(self.eval_dataset[t])  # 5.0 * len(self.eval_dataset[t])
        stop_buffer = []

        while agent_len.length < criterion_len and count < max_iter:
            if self.show_image and should_quit():
                status['exited'] = True
                break

            if frame is not None and self.agent.collision_sensor.has_collided(frame):
                logger.info('collision was detected at frame #{}'.format(frame))
                status['collided'] = True
                break

            if count > 30 and agent_len.length < 1:
                logger.info('simulation has a problem in going forward')
                status['exited'] = True
                break

            clock.tick()
            self.world.tick()
            try:
                ts = self.world.wait_for_tick()
            except RuntimeError as e:
                logger.error('runtime error: {}'.format(e))
                status['restart'] = True
                return status

            if frame is not None:
                if ts.frame_count != frame + 1:
                    logger.info('frame skip!')
            frame = ts.frame_count

            if self.agent.image_frame is None:
                continue
            if self.agent.segment_frame is None:
                continue

            # run high-level evaluator when stopped was triggered by the low-level controller
            final_image = self.final_image
            if status['stopped']:
                logger.info((final_image.shape))
                action = self.high_level_evaluator.run_step(final_image, sentence)
                action = self.softmax(action)
                logger.info((action, action.shape, sentence))
                action_index = torch.argmax(action[-1], dim=0).item()
                logger.info('action {}, action_index {}'.format(action, action_index))
                location = self.agent.fetch_car_state().transform.location
                self.high_level_data_dict[t].append((final_image, {
                    'sentence': sentence,
                    'location': (location.x, location.y),
                    'action_index': action_index}))
                if action_index < 4:
                    self.control_evaluator.cmd = action_index
                    self.stop_evaluator.cmd = action_index
                    stop_buffer = []
                else:
                    logger.info('the task was finished by "finish"')
                    status['finished'] = True
                    break

            # run low-level evaluator to apply control and update stopped status
            if count % EVAL_FRAMERATE_SCALE == 0:
                control: carla.VehicleControl = self.control_evaluator.run_step(final_image)
                stop: float = self.stop_evaluator.run_step(final_image)
                sub_goal = fetch_high_level_command_from_index(self.control_evaluator.cmd).lower()
                logger.info('throttle {:+6.4f}, steer {:+6.4f}, delayed {}, stop {:+6.4f}'.format(
                    control.throttle, control.steer, frame - self.agent.image_frame_number, stop))
                self.agent.step_from_control(frame, control)
                self.agent.save_stop(frame, stop, sub_goal)
                agent_len(self.agent.data_frame_dict[self.agent.data_frame_number].state.transform.location)
                stop_buffer.append(stop)
                recent_buffer = stop_buffer[-3:]
                status['stopped'] = len(recent_buffer) > 2 and sum(list(map(lambda x: x > 0.0, recent_buffer))) > 1

            if self.show_image and self.agent.image_frame is not None:
                self.show(self.agent.image_frame, clock)

            self.final_images.append(final_image)

            count += 1

            if agent_len.length >= criterion_len:
                logger.info('trajectory length is longer than the threshold')
            if count >= max_iter:
                logger.info('reached the maximum number of iterations')

        if not status['finished']:
            status['finished'] = status['collided'] or agent_len.length >= criterion_len or count >= max_iter
        if not status['finished']:
            return status
        curr_eval_data = self.agent.export_eval_data(status['collided'], sentence)
        if curr_eval_data is not None:
            status['saved'] = self.export_evaluation_data(t, curr_eval_data)
        return status

    def save_high_level_data(self):
        tmp_dir = mkdir_if_not_exists(Path.home() / '.tmp/high-level')
        for key in self.high_level_data_dict.keys():
            if len(self.high_level_data_dict[key]) != 4:
                continue
            data_dir = mkdir_if_not_exists(tmp_dir / '{:03d}'.format(key))
            dict_list = []
            for i, (image, item_dict) in enumerate(self.high_level_data_dict[key]):
                cv2.imwrite(str(data_dir / '{:03d}.png'.format(i)), image)
                dict_list.append(item_dict)
            with open(str(data_dir / 'data.json'), 'w') as file:
                json.dump(dict_list, file, indent=2)

    def run(self) -> bool:
        assert self.evaluation
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')
        if self.control_evaluator is None or self.stop_evaluator is None:
            raise ValueError('evluation call function was not set')

        old_indices = self.traj_indices_from_state_dir()
        exited = False
        while len(old_indices) < len(self.eval_transforms) and not exited:
            try:
                t = 0
                while t < len(self.eval_transforms):
                    if t in old_indices:
                        t += 1
                        continue
                    transform = self.eval_transforms[t]
                    run_status = self.run_single_trajectory(t, transform)
                    if run_status['finished']:
                        break
                    if run_status['restart']:
                        continue
                    if run_status['saved']:
                        old_indices.add(t)
                    t += 1
            finally:
                old_indices = self.traj_indices_from_state_dir()
        set_world_asynchronous(self.world)
        if self.agent is not None:
            self.agent.destroy()
        self.save_high_level_data()
        return True


class AllOnlineEvaluationEnvironment(GameEnvironment, EvaluationDirectory):
    def __init__(self, eval_keyword: str, args):
        self.eval_keyword = eval_keyword
        args.show_game = True
        GameEnvironment.__init__(self, args=args, agent_type='evaluation')

        # load params and evaluators
        self.eval_name = args.eval_name
        self.control_param, self.control_evaluator = load_param_and_evaluator(
            eval_keyword=eval_keyword, args=args, model_type='control')
        self.stop_param, self.stop_evaluator = load_param_and_evaluator(
            eval_keyword=eval_keyword, args=args, model_type='stop')
        self.high_param, self.high_evaluator = load_param_and_evaluator(
            eval_keyword=eval_keyword, args=args, model_type='high')

        # set image type
        self.image_type = self.high_param.image_type
        if 'd' in self.image_type:
            from model import DeepLabModel, prepare_deeplab_model
            self.deeplab_model: DeepLabModel = prepare_deeplab_model()

        self.final_images = []
        self.eval_dataset, self.eval_sentences = load_evaluation_dataset(self.high_param)
        self.eval_transforms = list(map(lambda x: x[0].state.transform, self.eval_dataset))
        self.high_sentences = self.eval_sentences
        self.softmax = torch.nn.Softmax(dim=1)
        EvaluationDirectory.__init__(self, *self.eval_info)
        self.high_data_dict = dict()

    @property
    def eval_info(self):
        return self.control_param.exp_index, self.eval_name, \
               self.control_evaluator.step, 'online'

    @property
    def segment_image(self):
        return np.reshape(((self.agent.segment_frame[:, :, 2] == 7).astype(dtype=np.uint8) * 255), (88, 200, 1))

    @property
    def custom_segment_image(self):
        return np.reshape(self.deeplab_model.run(self.agent.image_frame), (88, 200, 1))

    @property
    def final_image(self):
        if self.image_type == 's':
            return self.segment_image
        elif self.image_type == 'd':
            return self.custom_segment_image
        elif self.image_type == 'bgr':
            return self.agent.image_frame
        elif self.image_type == 'bgrs':
            return np.concatenate((self.agent.image_frame, self.segment_image), axis=-1)
        elif self.image_type == 'bgrd':
            return np.concatenate((self.agent.image_frame, self.custom_segment_image), axis=-1)
        else:
            raise TypeError('invalid image type {}'.format(self.image_type))

    def export_video(self, t: int, camera_keyword: str, curr_eval_data: dict):
        _, sub_goals = zip(*curr_eval_data['stop_frames'])
        texts = ['sentence: {}\nsub-task: {}'.format(s, g)
                 for g, s in zip(sub_goals, curr_eval_data['sentences'])]
        text_dict = {i: t for i, t in zip(range(*curr_eval_data['frame_range']), texts)}
        src_image_files = [self.agent.image_path(f, camera_keyword) for f in range(*curr_eval_data['frame_range'])]
        src_image_files = list(filter(lambda x: x.exists(), src_image_files))
        image_frames = set([int(s.stem[:-1]) for s in src_image_files])
        drive_frames = set(text_dict.keys())
        common_frames = sorted(list(image_frames.intersection(drive_frames)))
        src_image_files = [self.agent.image_path(f, camera_keyword) for f in common_frames]
        dst_image_files = [self.image_dir / p.name for p in src_image_files]
        [shutil.copy(str(s), str(d)) for s, d in zip(src_image_files, dst_image_files)]
        text_list = [text_dict[f] for f in common_frames]
        video_from_files(src_image_files, self.video_path(t, camera_keyword),
                         texts=text_list, framerate=30, revert=True)

    def export_segment_video(self, t: int):
        final_image_files = [self.segment_dir / '{:08d}.png'.format(i) for i in range(len(self.final_images))]
        logger.info('final_image_files {}'.format(len(final_image_files)))
        for p, s in zip(final_image_files, self.final_images):
            cv2.imwrite(str(p), s)
        video_from_files(final_image_files, self.video_dir / 'segment{:02d}.mp4'.format(t),
                         texts=[], framerate=30, revert=False)

    def export_evaluation_data(self, t: int, curr_eval_data: dict) -> bool:
        with open(str(self.state_path(t)), 'w') as file:
            json.dump(curr_eval_data, file, indent=2)

        data_frames = [DriveDataFrame.load_from_str(s) for s in curr_eval_data['data_frames']]
        controls = list(map(attrgetter('control'), data_frames))
        stops, sub_goals = zip(*curr_eval_data['stop_frames'])
        logger.info('controls, stops, goals {}, {}, {}'.format(len(controls), len(stops), len(sub_goals)))

        self.export_video(t, 'center', curr_eval_data)
        self.export_video(t, 'extra', curr_eval_data)
        self.export_segment_video(t)

        return self.state_path(t).exists()

    def run_single_trajectory(self, t: int, transform: carla.Transform) -> Dict[str, bool]:
        status = {
            'exited': False,  # has to finish the entire loop
            'finished': False,  # this procedure has been finished successfully
            'saved': False,  # successfully saved the evaluation data
            'collided': False,  # the agent has collided
            'restart': False,  # this has to be restarted
            'stopped': True  # low-level controller returns stop
        }
        self.agent.reset()
        self.agent.move_vehicle(transform)
        self.control_evaluator.initialize()
        self.stop_evaluator.initialize()
        self.high_evaluator.initialize()
        self.high_data_dict[t] = []
        self.final_images = []
        self.sentence = get_random_sentence_from_keyword(self.eval_keyword)
        logger.info('moved the vehicle to the position {}'.format(t))

        count = 0
        frame = None
        clock = pygame.time.Clock()

        set_world_asynchronous(self.world)
        sleep(0.5)
        set_world_synchronous(self.world)

        stop_buffer = []

        while not status['exited'] or not status['collided']:
            keyboard_input = listen_keyboard()
            if keyboard_input == 'q':
                status['exited'] = True
                break
            elif keyboard_input in __keyword_from_input__.keys():
                keyword = __keyword_from_input__[keyboard_input]
                if keyword != self.eval_keyword:
                    self.eval_keyword = keyword
                    self.sentence = get_random_sentence_from_keyword(self.eval_keyword)
                    self.control_param.eval_keyword = keyword
                    self.stop_param.eval_keyword = keyword
                    self.high_param.eval_keyword = keyword
                    self.control_evaluator.param = self.control_param
                    self.stop_evaluator.param = self.stop_param
                    self.high_evaluator.cmd = keyword
                    self.high_evaluator.param = self.high_param
                    self.high_evaluator.sentence = keyword.lower()
                    self.control_evaluator.initialize()
                    self.stop_evaluator.initialize()
                    self.high_evaluator.initialize()
                    logger.info('updated sentence {}'.format(self.sentence))

            if frame is not None and self.agent.collision_sensor.has_collided(frame):
                logger.info('collision was detected at frame #{}'.format(frame))
                status['collided'] = True
                break

            clock.tick()
            self.world.tick()
            try:
                ts = self.world.wait_for_tick()
            except RuntimeError as e:
                logger.error('runtime error: {}'.format(e))
                status['restart'] = True
                return status

            if frame is not None:
                if ts.frame_count != frame + 1:
                    logger.info('frame skip!')
            frame = ts.frame_count

            if self.agent.image_frame is None:
                continue
            if self.agent.segment_frame is None:
                continue

            # run high-level evaluator when stopped was triggered by the low-level controller
            final_image = self.final_image
            if status['stopped']:
                action = self.high_evaluator.run_step(final_image, self.sentence)
                action = self.softmax(action)
                action_index = torch.argmax(action[-1], dim=0).item()
                location = self.agent.fetch_car_state().transform.location
                self.high_data_dict[t].append((final_image, {
                    'sentence': self.sentence,
                    'location': (location.x, location.y),
                    'action_index': action_index}))
                if action_index < 4:
                    self.control_evaluator.cmd = action_index
                    self.stop_evaluator.cmd = action_index
                    stop_buffer = []
                else:
                    logger.info('the task was finished by "finish"')
                    status['finished'] = True
                    break

            # run low-level evaluator to apply control and update stopped status
            if count % EVAL_FRAMERATE_SCALE == 0:
                control: carla.VehicleControl = self.control_evaluator.run_step(final_image)
                stop: float = self.stop_evaluator.run_step(final_image)
                sub_goal = fetch_high_level_command_from_index(self.control_evaluator.cmd).lower()
                logger.info('throttle {:+6.4f}, steer {:+6.4f}, delayed {}, current {:d}, stop {:+6.4f}'.
                            format(control.throttle, control.steer, frame - self.agent.image_frame_number, action_index,
                                   stop))
                self.agent.step_from_control(frame, control)
                self.agent.save_stop(frame, stop, sub_goal)
                self.agent.save_cmd(frame, self.sentence)
                stop_buffer.append(stop)
                recent_buffer = stop_buffer[-3:]
                status['stopped'] = len(recent_buffer) > 2 and sum(list(map(lambda x: x > 0.0, recent_buffer))) > 1

            if self.show_image and self.agent.image_frame is not None:
                self.show(self.agent.image_frame, clock, extra_str=self.sentence)

            self.final_images.append(final_image)

            count += 1
        logger.info('saving information')
        curr_eval_data = self.agent.export_eval_data(status['collided'], self.sentence)
        if curr_eval_data is not None:
            status['saved'] = self.export_evaluation_data(t, curr_eval_data)
        return status

    def run(self) -> bool:
        assert self.evaluation
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')
        if self.control_evaluator is None or self.stop_evaluator is None:
            raise ValueError('evaluation call function was not set')

        old_indices = self.traj_indices_from_state_dir()
        exited = False
        while len(old_indices) < len(self.eval_transforms) and not exited:
            try:
                t = 0
                while t < len(self.eval_transforms):
                    if t in old_indices:
                        t += 1
                        continue
                    transform = self.eval_transforms[t]
                    run_status = self.run_single_trajectory(t, transform)
                    if run_status['exited']:
                        exited = True
                        break
                    if run_status['finished']:
                        break
                    if run_status['restart']:
                        continue
                    if run_status['saved']:
                        old_indices.add(t)
                    t += 1
            finally:
                old_indices = self.traj_indices_from_state_dir()
        set_world_asynchronous(self.world)
        if self.agent is not None:
            self.agent.destroy()
        return not exited