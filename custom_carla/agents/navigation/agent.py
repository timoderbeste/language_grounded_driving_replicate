#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """
import json
import math
import random

from enum import Enum
from functools import partial
from typing import List, Tuple

import carla
import numpy as np

from custom_carla.agents.navigation.basic_agent import BasicAgent
from custom_carla.agents.navigation.local_planner import RoadOption
from custom_carla.agents.navigation.roaming_agent import RoamingAgent
from custom_carla.agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle
from data.types import CarState, DriveDataFrame, CarControl
from game_environment import set_world_rendering_option, set_world_asynchronous, set_world_synchronous, logger, destroy_actor
from sensor import CameraSensor, SegmentationSensor, CollisionSensor
from util.common import get_timestamp
from util.directory import ExperimentDirectory
from util.serialize import str_from_waypoint


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3


class WaypointModifiableLocation:
    def __init__(self, waypoint):
        if isinstance(waypoint, carla.Waypoint):
            self.waypoint = waypoint
            self.location_ = carla.Location(
                waypoint.transform.location.x,
                waypoint.transform.location.y,
                waypoint.transform.location.z)
        elif isinstance(waypoint, WaypointModifiableLocation):
            self.waypoint = waypoint.waypoint
            self.location_ = carla.Location(waypoint.x, waypoint.y, waypoint.z)
        else:
            raise TypeError('invalid type was given {}'.format(type(waypoint)))

    @property
    def location(self):
        return self.location_

    @location.setter
    def location(self, l: carla.Location):
        self.location_ = l

    @property
    def x(self):
        return self.location.x

    @x.setter
    def x(self, nx):
        self.location_.x = nx

    @property
    def y(self):
        return self.location.y

    @y.setter
    def y(self, ny):
        self.location_.y = ny

    @property
    def z(self):
        return self.location.z

    @z.setter
    def z(self, nz):
        self.location_.z = nz

    @property
    def id(self):
        return self.waypoint.id

    def next(self, value):
        return [WaypointModifiableLocation(w) for w in self.waypoint.next(value)]


class WaypointWithInfo:
    def __init__(
        self,
        waypoint: WaypointModifiableLocation,
        road_option=None,
        possible_road_options=[]):
        self.waypoint = waypoint
        self.road_option = road_option
        self.possible_road_options = possible_road_options


class ControlWithInfo:
    def __init__(
            self,
            control: carla.VehicleControl = carla.VehicleControl(),
            waypoint: carla.Waypoint = None,
            road_option=None,
            possible_road_options=[],
            waypoint_id=-1):
        self.control = control
        self.waypoint = waypoint
        self.road_option = road_option
        self.possible_road_options = possible_road_options
        self.waypoint_id_ = waypoint_id
        self.is_emergency_ = False

    @property
    def valid(self):
        return self.waypoint_id >= 0 and self.road_option is not None and not self.possible_road_options

    @property
    def is_emergency(self):
        return self.is_emergency_ and not self.valid

    @property
    def has_waypoint(self):
        return self.waypoint is not None

    @staticmethod
    def emergency_stop():
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control_info = ControlWithInfo(control=control)
        control_info.is_emergency_ = True
        return control_info

    @property
    def waypoint_id(self):
        if self.waypoint_id_ >= 0:
            return self.waypoint_id_
        return -1 if self.waypoint is None else self.waypoint.id


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._proximity_threshold = 10.0  # meters
        self._local_planner = None
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._last_traffic_light = None
        self._state = AgentState.NAVIGATING

    def restart(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        if self._map.name == 'Town01' or self._map.name == 'Town02':
            return self._is_light_red_europe_style(lights_list)
        else:
            return self._is_light_red_us_style(lights_list)

    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                  affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_waypoint = self._map.get_waypoint(traffic_light.get_location())
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = traffic_light.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _is_light_red_us_style(self, lights_list, debug=False):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_intersection:
            # It is too late. Do not block the intersection! Keep going!
            return (False, None)

        if self._local_planner.target_waypoint is not None:
            if self._local_planner.target_waypoint.is_intersection:
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 60.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                            sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return (True, self._last_traffic_light)
                else:
                    self._last_traffic_light = None

        return (False, None)

    def _is_vehicle_hazard(self, vehicle_list):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()
            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        self._proximity_threshold):
                return (True, target_vehicle)

        return (False, None)

    def run_step(self, debug=False) -> ControlWithInfo:
        """
        Execute one step of navigation.
        :return: ControlWithInfo
        """
        if self._local_planner is None:
            return ControlWithInfo()

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            return ControlWithInfo.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            return self._local_planner.run_step(debug)


class SynchronousAgent(ExperimentDirectory):
    def __init__(
            self,
            world: carla.World,
            args,
            transform: carla.Transform,
            agent_type: str,
            render_image: bool,
            evaluation: bool):
        self.world = world
        self.map = world.get_map()
        self.vehicle = None
        self.agent = None

        self.camera_sensor_dict = None
        self.segment_sensor_dict = None
        self.collision_sensor = None

        self.args = args
        self.agent_type = agent_type
        self.render_image = render_image
        self.evaluation = evaluation

        self.image_width = args.width
        self.image_height = args.height
        self.camera_keywords: List[str] = args.camera_keywords

        set_world_rendering_option(self.world, self.render_image)

        self.data_frame_dict = dict()
        self.data_frame_number_ = None
        self.progress_index = 0
        self.data_frame_buffer = set()
        self.stop_dict = dict()
        self.cmd_dict = dict()

        ExperimentDirectory.__init__(self, get_timestamp())
        self.target_waypoint_ = None
        self.waypoint_dict = dict()
        self.waypoint_buffer = set()

        set_world_asynchronous(self.world)

        self.set_vehicle(transform)
        if self.autopilot:
            self.set_agent()
        self.export_meta()

        set_world_synchronous(self.world)
        if self.render_image:
            self.set_camera_sensor()
            self.set_segment_sensor()
        self.set_collision_sensor()

    def reset(self):
        self.data_frame_dict = dict()
        self.data_frame_number_ = None
        self.progress_index = 0
        self.data_frame_buffer = set()
        if self.camera_sensor_dict is not None:
            for keyword in self.camera_sensor_dict.keys():
                self.camera_sensor_dict[keyword].image_frame_number = None
                self.camera_sensor_dict[keyword].image_frame = None
        if self.segment_sensor_dict is not None:
            for keyword in self.segment_sensor_dict.keys():
                self.segment_sensor_dict[keyword].image_frame_number = None
                self.segment_sensor_dict[keyword].image_frame = None
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        self.vehicle.apply_control(control)

    @property
    def autopilot(self) -> bool:
        return self.agent_type in ['roaming', 'basic']

    @property
    def image_frame_number(self):
        return self.camera_sensor_dict['center'].image_frame_number

    @property
    def image_frame(self):
        return self.camera_sensor_dict['center'].image_frame

    @property
    def segment_frame_number(self):
        return self.segment_sensor_dict['center'].image_frame_number

    @property
    def segment_frame(self):
        return self.segment_sensor_dict['center'].image_frame

    @property
    def route(self):
        if not self.autopilot:
            raise ValueError('autopilot was not set')
        return self.agent._route

    def set_vehicle(self, transform):
        blueprints = self.world.get_blueprint_library().filter('vehicle.audi.a2')
        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        if self.vehicle is not None and self.agent is not None:
            return

        blueprint_vehicle = random.choice(blueprints)
        blueprint_vehicle.set_attribute('role_name', 'hero')
        if blueprint_vehicle.has_attribute('color'):
            color = random.choice(blueprint_vehicle.get_attribute('color').recommended_values)
            blueprint_vehicle.set_attribute('color', color)
        blueprint_vehicle.set_attribute('role_name', 'autopilot')

        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(blueprint_vehicle, transform)
        self.vehicle.set_autopilot(False)

    def move_vehicle(self, transform: carla.Transform):
        transform.location.z += 0.1
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_transform(transform)
        self.vehicle.set_simulate_physics(True)

    def set_destination(self, src, dst):
        if self.agent is not None and dst is not None:
            self.agent.set_destination(src, dst)

    def set_agent(self):
        if self.vehicle is None:
            raise ValueError('vehicle is not assigned')
        if self.agent_type == 'roaming':
            self.agent = RoamingAgent(self.vehicle, self.args.speed)
        elif self.agent_type == 'basic':
            self.agent = BasicAgent(self.vehicle, self.args.speed)
        else:
            raise TypeError('invalid agent type: {}'.format(self.agent_type))

    def set_camera_sensor(self):
        self.camera_sensor_dict = {
            camera_keyword: CameraSensor(
                self.vehicle,
                partial(self.image_path, camera_keyword=camera_keyword),
                self.image_width,
                self.image_height,
                camera_keyword)
            for camera_keyword in self.camera_keywords}
        self.camera_sensor_dict['extra'] = CameraSensor(
            self.vehicle, partial(self.image_path, camera_keyword='extra'),
            640, 480, 'extra')

    def set_segment_sensor(self):
        self.segment_sensor_dict = {
            camera_keyword: SegmentationSensor(
                self.vehicle,
                partial(self.segment_image_path, camera_keyword=camera_keyword),
                self.image_width,
                self.image_height,
                camera_keyword)
            for camera_keyword in self.camera_keywords}

    def set_collision_sensor(self):
        self.collision_sensor = CollisionSensor(self.vehicle)

    @property
    def data_frame_number(self):
        return self.data_frame_number_

    @data_frame_number.setter
    def data_frame_number(self, frame: int):
        if self.data_frame_number is None or self.data_frame_number < frame:
            self.data_frame_buffer.add(frame)
            self.data_frame_number_ = frame

    @property
    def target_waypoint(self):
        return self.target_waypoint_

    @target_waypoint.setter
    def target_waypoint(self, waypoint: carla.Waypoint):
        if waypoint is None:
            return
        if waypoint.id not in self.waypoint_dict:
            self.waypoint_buffer.add(waypoint.id)
            self.waypoint_dict[waypoint.id] = waypoint
        self.target_waypoint_ = waypoint

    def fetch_image_frame(self, camera_keyword: str) -> Tuple[int, np.ndarray]:
        if camera_keyword not in self.camera_sensor_dict:
            raise KeyError('invalid camera index {}'.format(camera_keyword))
        return self.camera_sensor_dict[camera_keyword].image_frame_number, \
               self.camera_sensor_dict[camera_keyword].image_frame

    def fetch_car_state(self) -> CarState:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_velocity = self.vehicle.get_angular_velocity()
        acceleration = self.vehicle.get_acceleration()
        return CarState(transform, velocity, angular_velocity, acceleration)

    def save_stop(self, frame: int, stop: float, sub_goal: str):
        if stop is not None:
            self.stop_dict[frame] = stop, sub_goal

    def save_cmd(self, frame: int, action_values: List[str]):
        self.cmd_dict[frame] = action_values

    def step_from_pilot(
            self,
            frame: int,
            apply: bool = True,
            update: bool = True,
            inject: float = 0.0) -> Tuple[carla.Waypoint, RoadOption, DriveDataFrame]:
        control_with_info: ControlWithInfo = self.agent.run_step(debug=False)
        vehicle_control: carla.VehicleControl = control_with_info.control
        vehicle_control.manual_gear_shift = False
        car_state = self.fetch_car_state()
        drive_data_frame = DriveDataFrame(car_state, control_with_info)
        velocity = car_state.velocity
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if apply:
            vehicle_control.steer += inject
            # if abs(inject) > 1e-3 and abs(vehicle_control.steer - control_with_info.control.steer) < 1e-3:
            #     logger.error('failed to inject noise')
            # print('{:+4.2f}, {:+4.2f}'.format(vehicle_control.throttle, vehicle_control.steer))
            self.vehicle.apply_control(vehicle_control)

        if update:
            self.data_frame_number = frame
            self.data_frame_dict[self.data_frame_number] = drive_data_frame
            self.target_waypoint = control_with_info.waypoint
            # assert control_with_info.has_waypoint
        return control_with_info.waypoint, control_with_info.road_option, drive_data_frame

    def step_from_control(
            self,
            frame: int,
            vehicle_control: carla.VehicleControl,
            apply: bool = True,
            update: bool = True) -> None:
        throttle_value = vehicle_control.throttle
        if apply:
            vehicle_control.manual_gear_shift = False
            if throttle_value < 0.4:
                vehicle_control.throttle = 0.4  # avoid stopping
            # todo: implement PID controller
            if self.data_frame_number is not None and self.data_frame_number in self.data_frame_dict:
                velocity = self.data_frame_dict[self.data_frame_number].state.velocity
                speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                logger.info('speed {:+5.3f}'.format(speed))
                if speed > 20:
                    vehicle_control.throttle = 0.0
            self.vehicle.apply_control(vehicle_control)
        if update:
            car_control = CarControl.load_from_vehicle_control(vehicle_control)
            car_control.throttle = throttle_value
            control_with_info = ControlWithInfo(control=car_control, road_option=RoadOption.VOID)
            car_state = self.fetch_car_state()
            drive_data_frame = DriveDataFrame(car_state, control_with_info)
            self.data_frame_number = frame
            self.data_frame_dict[self.data_frame_number] = drive_data_frame

    def destroy(self):
        destroy_actor(self.vehicle)

    def __del__(self):
        self.destroy()

    def export_meta(self):
        meta = {
            'world_id': self.world.id,
            'map_name': self.world.get_map().name,
            'vehicle_id': self.vehicle.id,
            'vehicle_type_id': self.vehicle.type_id,
            'vehicle_color': self.vehicle.attributes['color']
        }
        with open(str(self.experiment_meta_path), 'w') as file:
            json.dump(meta, file, indent=4)

    def export_data(self):
        logger.info('export data: {} data frames, {} waypoints'.format(
            len(self.data_frame_buffer), len(self.waypoint_buffer)))
        frame_strs = []
        data_frame_numbers = sorted(self.data_frame_buffer)
        for data_frame_number in data_frame_numbers:
            frame_strs.append('{}:{}'.format(self.frame_str(data_frame_number), self.data_frame_dict[data_frame_number]))
        with open(str(self.experiment_data_path), 'a') as file:
            file.write('\n'.join(frame_strs) + '\n')
        self.data_frame_buffer.clear()

        waypoint_strs = []
        for waypoint_id in self.waypoint_buffer:
            waypoint_strs.append(str_from_waypoint(self.waypoint_dict[waypoint_id]))
        with open(str(self.experiment_waypoint_path), 'a') as file:
            file.write('\n'.join(waypoint_strs) + '\n')
        self.waypoint_buffer.clear()

    def export_eval_data(self, collided: bool, sentence) -> dict:
        logger.info('export data: {} data frames'.format(len(self.data_frame_buffer)))
        data_frame_numbers = sorted(self.data_frame_buffer)
        if not data_frame_numbers:
            return None
        data_frame_range = data_frame_numbers[0], data_frame_numbers[-1] + 1
        data_frame_strs = [str(self.data_frame_dict[f]) for f in data_frame_numbers]
        stop_sub_goal_lists = [self.stop_dict[f] if f in self.stop_dict else (0.0, 'None') for f in data_frame_numbers]
        if list(self.cmd_dict.keys()):
            sentences = [self.cmd_dict[f] for f in data_frame_numbers]
        else:
            sentences = []
        self.data_frame_buffer.clear()
        return {
            'sentence': sentence,
            'frame_range': data_frame_range,
            'collided': collided,
            'data_frames': data_frame_strs,
            'stop_frames': stop_sub_goal_lists,
            'sentences': sentences
        }