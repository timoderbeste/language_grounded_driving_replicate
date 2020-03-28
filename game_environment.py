import math
import random
from operator import attrgetter, itemgetter
from time import perf_counter, sleep
from typing import Tuple, List, Dict

import cv2

from custom_carla.agents.navigation.agent import SynchronousAgent
from config import EVAL_FRAMERATE_SCALE, CAMERA_KEYWORDS
from data.storage import DataStorage
from data.types import LengthComputer
from data_generator import logger, align_indices_from_dicts
from evaluator import LowLevelEvaluator
from parameter import Parameter
from util.common import add_carla_module, get_logger, set_random_seed
from util.directory import fetch_dataset_dir
from util.road_option import fetch_name_from_road_option

add_carla_module()
logger = get_logger(__name__)
import carla
import numpy as np
import pygame


SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


def destroy_actor(actor):
    if actor is not None and actor.is_alive:
        actor.destroy()


class FrameCounter:
    def __init__(self):
        self.t1 = None
        self.t2 = None
        self.counter = 0

    def tick(self):
        if self.t1 is None:
            self.t1 = perf_counter()
        self.t2 = perf_counter()
        self.counter += 1

    @property
    def framerate(self):
        if self.counter == 0 or self.t1 is None or self.t2 is None or self.t1 == self.t2:
            return 0.0
        else:
            return self.counter / (self.t2 - self.t1)

    def reset(self):
        self.t1 = None
        self.t2 = None
        self.counter = 0

    def __str__(self):
        return 'count: {:5d}, fps: {:4.2f}'.format(self.counter, self.framerate)


def draw_image(surface, image: np.ndarray):
    array = image[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def keyboard_control() -> int:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 2  # break the outer loop
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return 2
            elif event.key in [pygame.K_n, pygame.K_q]:
                return 1  # continue to next iteration
            elif event.key == pygame.K_1:
                return 11
            elif event.key == pygame.K_2:
                return 12
            elif event.key == pygame.K_3:
                return 13
            elif event.key == pygame.K_4:
                return 14
    return 0  # nothing happened


def waypoint_info(display, font, waypoint):
    """abcd"""
    def render(font, key, value):
        return font.render('{}:{}'.format(key, value), True, (255, 255, 255))

    px, py, hy = 8, 20, 10
    keys = ['lane_type', 'lane_change', 'is_intersection', 'road_id', 'section_id', 'lane_id', 's']
    values = [(key, attrgetter(key)(waypoint)) for key in keys]
    texts = []
    nexts = waypoint.next(2)
    texts.append(render(font, 'next', len(nexts)))
    for key_value_pair in values:
        key, value = key_value_pair
        if isinstance(value, carla.LaneMarking):
            value = value.type
        texts.append(render(font, key, value))

    heights = list(range(py, py + hy * (len(texts) - 1), hy))
    for text, height in zip(texts, heights):
        display.blit(text, (px, height))


def get_synchronous_mode(world: carla.World):
    return world.get_settings().synchronous_mode


def set_world_synchronous(world: carla.World):
    if world is None:
        return
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        world.apply_settings(settings)


def set_world_asynchronous(world: carla.World):
    if world is None:
        return
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        world.apply_settings(settings)


def set_world_rendering_option(world: carla.World, render: bool):
    if world is None:
        return
    settings = world.get_settings()
    settings.no_rendering_mode = not render
    world.apply_settings(settings)


def set_traffic_lights_green(world: carla.World):
    set_world_asynchronous(world)
    for tl in world.get_actors().filter('traffic.*'):
        if isinstance(tl, carla.TrafficLight):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)


def clean_vehicles(world):
    set_world_asynchronous(world)
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.is_alive:
            actor.destroy()


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def numpy_from_carla_image(carla_image: carla.Image) -> np.ndarray:
    np_image = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape(carla_image.height, carla_image.width, 4)
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
    return rgb_image


CAMERA_SHIFT = 0.4
__camera_transforms__ = {
    'center': carla.Transform(carla.Location(x=1.6, z=1.7)),
    'left': carla.Transform(
        carla.Location(x=1.6, y=-CAMERA_SHIFT, z=1.7),
        carla.Rotation(yaw=math.atan2(-CAMERA_SHIFT, 1.6) * 180 / math.pi)),
    'right': carla.Transform(
        carla.Location(x=1.6, y=+CAMERA_SHIFT, z=1.7),
        carla.Rotation(yaw=math.atan2(CAMERA_SHIFT, 1.6) * 180 / math.pi)),
    'extra': carla.Transform(carla.Location(x=1.6, z=1.7))
}
for key in __camera_transforms__.keys():
    assert key in CAMERA_KEYWORDS
for key in CAMERA_KEYWORDS:
    assert key in __camera_transforms__


def show_game(
        display,
        font,
        image,
        clock,
        road_option = None,
        is_intersection = None,
        extra_str: str = ''):
    draw_image(display, image)
    strs = ['{:5.3f}'.format(clock.get_fps())]
    if road_option is not None:
        strs += [road_option.name.lower()]
    if is_intersection is not None:
        strs += [str(is_intersection)]
    if extra_str:
        strs += [extra_str]
    text_surface = font.render(', '.join(strs), True, (255, 255, 255))
    display.blit(text_surface, (8, 10))
    pygame.display.flip()


class GameEnvironment:
    def __init__(self, args, agent_type: str, transform_index: int = 0):
        self.args = args
        self.transform_index = transform_index

        set_random_seed(0)

        self.agent_type = agent_type
        self.evaluation = True if agent_type == 'evaluation' else False
        self.client = None
        self.world = None
        self.agent = None

        self.display = None
        self.font = None
        self.render_image = not args.no_rendering
        self.show_image = args.show_game if self.render_image else False

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.transforms = self.world.get_map().get_spawn_points()
        random.shuffle(self.transforms)

        if self.show_image:
            import pygame
            pygame.init()
            self.display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()

        set_world_asynchronous(self.world)
        clean_vehicles(self.world)
        set_traffic_lights_green(self.world)
        self.agent = SynchronousAgent(
            world=self.world,
            args=self.args,
            transform=self.transforms[self.transform_index],
            agent_type=self.agent_type,
            render_image=self.render_image,
            evaluation=self.evaluation)
        assert self.world.get_settings().synchronous_mode

    @property
    def transform_index(self):
        return self.transform_index_ % len(self.transforms)

    @transform_index.setter
    def transform_index(self, value: int):
        self.transform_index_ = value

    def show(self, image, clock, road_option = None, is_intersection = None, extra_str: str = ''):
        assert self.show_image
        show_game(self.display, self.font, image, clock, road_option, is_intersection, extra_str)

    def run(self) -> None:
        raise NotImplementedError


class DaggerGeneratorEnvironment(GameEnvironment):
    def __init__(self, args, eval_param: Parameter, index_data_list: List[Tuple[int, dict]]):
        assert eval_param.eval_data_name
        GameEnvironment.__init__(self, args=args, agent_type='basic')

        self.eval_param = eval_param
        self.evaluator = LowLevelEvaluator(self.eval_param, 0)
        self.evaluator.load(step=args.exp_step)
        self.index_data_list = index_data_list
        logger.info('dagger data indices: {}'.format(list(map(itemgetter(0), index_data_list))))
        self.dataset_name = eval_param.eval_data_name
        self.dataset = DataStorage(False, fetch_dataset_dir() / self.dataset_name)
        self.dagger_segments = []

    @property
    def eval_info(self):
        return self.eval_param.exp_index, self.eval_param.exp_name, self.evaluator.step, self.eval_param.eval_keyword

    def run_single_trajectory(self, t: int, data: dict) -> Dict[str, bool]:
        status = {
            'exited': False,  # has to finish the entire loop
            'finished': False,  # this procedure has been finished successfully
            'collided': False,  # the agent has collided
            'restart': False  # this has to be restarted
        }
        self.evaluator.cmd = data['action_index']
        self.agent.reset()
        logger.info('moved the vehicle to the position {}, set action to {}'.format(t, data['action_index']))

        local_image_dict = dict()
        local_drive_dict = dict()

        count = 0
        frame = None
        clock = pygame.time.Clock() if self.show_image else FrameCounter()
        # todo: implement this function as in the same one in evaluator.py

        set_world_asynchronous(self.world)
        self.agent.agent.set_destination(data['src_transform'].location, data['dst_location'])
        self.agent.move_vehicle(data['src_transform'])
        sleep(0.5)
        set_world_synchronous(self.world)

        len_waypoints = LengthComputer()
        for waypoint_with_info in self.agent.agent._route:
            len_waypoints(waypoint_with_info.waypoint.transform.location)
        max_len = 0.9 * len_waypoints.length
        max_iter = 5.0 * EVAL_FRAMERATE_SCALE * len(self.agent.agent._route)
        len_agent = LengthComputer()
        while count < max_iter and len_agent.length < max_len:
            if self.show_image and should_quit():
                status['exited'] = True
                return status

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

            # image_frame_number, image = self.agent.image_queue.get()
            if self.agent.image_frame is None:
                continue

            # register images
            image = self.agent.image_frame
            image_frame_number = self.agent.image_frame_number
            local_image_dict[image_frame_number] = image

            # store action values from the expert
            waypoint, road_option, drive_data_frame = self.agent.step_from_pilot(frame, apply=False, update=True)
            local_drive_dict[frame] = self.agent.data_frame_dict[frame]
            if waypoint is None:
                status['finished'] = True
                break

            # apply action values from the agent
            if count % EVAL_FRAMERATE_SCALE == 0:
                model_control = self.evaluator.run_step(self.agent.image_frame)
                control_str = 'throttle {:+6.4f}, steer {:+6.4f}, delayed {}'.format(
                    model_control.throttle, model_control.steer, frame - image_frame_number)
                if image_frame_number in local_drive_dict:
                    expert_control = local_drive_dict[image_frame_number].control
                    control_str += ' steer {:+6.4f}, steer-diff {:+6.4f}'.format(
                        expert_control.steer, model_control.steer - expert_control.steer)
                logger.info(control_str)
                self.agent.step_from_control(frame, model_control, apply=True, update=False)
                len_agent(drive_data_frame.state.transform.location)

            if self.show_image:
                self.show(image, clock)

            count += 1

        aligned, image_indices, drive_indices = align_indices_from_dicts(
            local_image_dict, local_drive_dict, EVAL_FRAMERATE_SCALE // 2)
        if aligned:
            road_option_name = fetch_name_from_road_option(data['road_option'])
            self.dataset.put_data_from_dagger(
                t, road_option_name, local_image_dict, local_drive_dict, image_indices, drive_indices)
            logger.info('successfully added {} dagger trajectory'.format(t))
        else:
            status['restart'] = True
        return status

    def run(self):
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')
        if self.evaluator is None:
            raise ValueError('evluation call function was not set')

        try:
            i = 0
            while i < len(self.index_data_list):
                index, data = self.index_data_list[i]
                if not self.dataset.has_trajectory(index):
                    run_status = self.run_single_trajectory(index, data)
                    if run_status['exited']:
                        break
                    if run_status['restart']:
                        continue
                i += 1
        finally:
            set_world_asynchronous(self.world)
            if self.agent is not None:
                self.agent.destroy()


class OfflineGeneratorEnvironment(GameEnvironment):
    def __init__(self, args):
        GameEnvironment.__init__(self, args=args, agent_type='roaming', transform_index=0)

    def run(self):
        if self.world is None:
            raise ValueError('world was not initialized')
        if self.agent is None:
            raise ValueError('agent was not initialized')

        try:
            frame = None
            count = 0
            clock = pygame.time.Clock() if self.show_image else FrameCounter()

            set_world_asynchronous(self.world)
            sleep(0.5)
            set_world_synchronous(self.world)

            waypoint_dict = dict()
            road_option_dict = dict()

            while count < 200000:
                if self.show_image and should_quit():
                    break

                if frame is not None and self.agent.collision_sensor.has_collided(frame):
                    logger.info('collision was detected at frame #{}'.format(frame))
                    set_world_asynchronous(self.world)
                    self.transform_index += 1
                    self.agent.move_vehicle(self.transforms[self.transform_index])
                    sleep(0.5)
                    self.agent.agent.reset_planner()
                    set_world_synchronous(self.world)

                clock.tick()
                self.world.tick()
                ts = self.world.wait_for_tick()
                if frame is not None:
                    if ts.frame_count != frame + 1:
                        logger.info('frame skip!')
                frame = ts.frame_count

                if len(self.agent.data_frame_buffer) > 100:
                    self.agent.export_data()
                    if not self.show_image:
                        print(str(clock))

                if self.agent.image_frame is None:
                    continue

                waypoint, road_option, _ = self.agent.step_from_pilot(
                    frame, update=True, apply=True, inject=0.0)
                waypoint_dict[frame] = waypoint
                road_option_dict[frame] = road_option
                if self.show_image:
                    image = self.agent.image_frame
                    image_frame_number = self.agent.image_frame_number
                    image_road_option = road_option_dict[image_frame_number] if image_frame_number in road_option_dict else None
                    image_waypoint = waypoint_dict[image_frame_number] if image_frame_number in waypoint_dict else None
                    self.show(image, clock, image_road_option, image_waypoint.is_intersection if image_waypoint is not None else None)

                count += 1
            self.agent.export_data()

        finally:
            set_world_asynchronous(self.world)
            if self.agent is not None:
                self.agent.destroy()