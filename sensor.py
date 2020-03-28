import math
import weakref
from collections import defaultdict

import cv2

from config import DATASET_FRAMERATE, EVAL_FRAMERATE_SCALE
from game_environment import destroy_actor, __camera_transforms__, numpy_from_carla_image


class SensorBase:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.world = self._parent.get_world()
        self.sensor = self.generate_sensor()

    def generate_sensor(self):
        raise NotImplementedError

    def destroy(self):
        destroy_actor(self.sensor)

    def __del__(self):
        self.destroy()


class CollisionSensor(SensorBase):
    def __init__(self, parent_actor):
        SensorBase.__init__(self, parent_actor)
        self.history = defaultdict(int)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        return self.world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

    def has_collided(self, frame_number: int) -> bool:
        return self.history[frame_number] > 0

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        # actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history[event.frame_number] += intensity


class CameraSensor(SensorBase):
    def __init__(self, parent_actor, image_path_func, width, height, camera_keyword: str):
        self.width = width
        self.height = height
        self.camera_keyword = camera_keyword
        SensorBase.__init__(self, parent_actor)
        self.image_path_func = image_path_func
        self.image_frame_number = None
        self.image_frame = None
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor.on_listen(weak_self, image))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
        return self.world.spawn_actor(bp, __camera_transforms__[self.camera_keyword], attach_to=self._parent)

    @staticmethod
    def on_listen(weak_self, carla_image: carla.Image):
        self = weak_self()
        if not self:
            return
        frame_number = carla_image.frame_number
        self.image_frame_number = frame_number
        numpy_image = numpy_from_carla_image(carla_image)
        self.image_frame = numpy_image
        cv2.imwrite(str(self.image_path_func(frame_number)), numpy_image)


class SegmentationSensor(SensorBase):
    def __init__(self, parent_actor, image_path_func, width, height, camera_keyword: str):
        self.width = width
        self.height = height
        self.camera_keyword = camera_keyword
        SensorBase.__init__(self, parent_actor)
        self.image_path_func = image_path_func
        self.image_frame_number = None
        self.image_frame = None
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: SegmentationSensor.on_listen(weak_self, image))

    def generate_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('sensor_tick', str(1 / (DATASET_FRAMERATE * EVAL_FRAMERATE_SCALE)))
        return self.world.spawn_actor(bp, __camera_transforms__[self.camera_keyword], attach_to=self._parent)

    @staticmethod
    def on_listen(weak_self, carla_image: carla.Image):
        self = weak_self()
        if not self:
            return
        frame_number = carla_image.frame_number
        self.image_frame_number = frame_number
        numpy_image = numpy_from_carla_image(carla_image)
        self.image_frame = numpy_image
        cv2.imwrite(str(self.image_path_func(frame_number)), numpy_image)