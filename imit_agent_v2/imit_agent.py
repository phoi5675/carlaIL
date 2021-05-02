#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import tensorflow as tf

import numpy as np
from PIL import Image
import os
import sys
import math
import random
import cv2
import time
from network import Network
import tensorflow_yolov3.carla.utils as utils


# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption


class ImitAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20, image_cut=[115, 510]):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(ImitAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0 / 20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict})
        self._hop_resolution = 1.0
        self._path_seperation_hop = 3
        self._path_seperation_threshold = 1.0
        self._target_speed = target_speed
        self._grp = None

        # data from vehicle
        self._speed = 0
        self._radar_data = None
        self._obstacle_ahead = False

        # load network
        g1 = tf.Graph()
        g2 = tf.Graph()

        with g1.as_default():
            self.drive_network = Network(model_name='Network', model_dir='/model_drive/')
        with g2.as_default():
            self.lanechange_network = Network(model_name='Lanechange', model_dir='/model_lanechange/')

        self._image_cut = image_cut
        self._image_size = (88, 200, 3)  # 아마 [세로, 가로, 차원(RGB)] 인듯?

        self.front_image = None

        # traffic light detection
        self.return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                                "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
        path = os.path.dirname(os.path.abspath(__file__))
        self.pb_file = os.path.join(path, "tensorflow_yolov3/yolov3_coco.pb")
        self.num_classes = 5
        self.traffic_image_input_size = 256
        self.tf_graph = tf.Graph()
        self.return_tensors = utils.read_pb_return_tensors(self.tf_graph, self.pb_file, self.return_elements)
        self.traffic_light_image = None
        self._is_traffic_light_in_distance = False
        self.bounding_boxes = None
        self.traffic_light_duration = 1.5
        self.traffic_light_detected_time = 0.0
        self.traffic_sess = tf.Session(graph=self.tf_graph)

    def set_destination(self, location, start_loc=None):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        if start_loc is not None:
            start_waypoint = self._map.get_waypoint(carla.Location(start_loc[0], start_loc[1], start_loc[2]))
        else:
            start_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

        self._local_planner.change_intersection_hcl(enter_hcl_len=5, exit_hcl_len=7)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        self._state = AgentState.NAVIGATING
        # standard local planner behavior
        self._local_planner.buffer_waypoints()

        direction = self.get_high_level_command(convert=False)
        v = self._vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)  # use m/s
        self._speed = speed * 3.6  # use km/s

        control = self._compute_action(self.front_image, speed, direction)
        return control

    def _compute_action(self, rgb_image, speed, direction=None):
        """
        Calculate steer, gas, brake from image input
        :return: carla.VehicleControl
        """
        '''
        # TODO scipy 제대로 되는지 확인
        # scipy 에서 imresize 가 depreciated 됐으므로 다른 방법으로 이미지 리사이즈
        # 이미지를 비율에 어느 정도 맞게 크롭 (395 * 800)
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        # 크롭한 이미지를 리사이징. 비율에 맞게 조절하는게 아니라 조금 찌그러지게 리사이징함. 원래 비율은 352 * 800
        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])
        '''

        rgb_image.convert(cc.Raw)

        array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
        array = array[self._image_cut[0]:self._image_cut[1], :, :3]  # 필요 없는 부분을 잘라내고
        array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        image_pil = image_pil.resize((self._image_size[1], self._image_size[0]))  # 원하는 크기로 리사이즈
        image_input = np.array(image_pil, dtype=np.dtype("uint8"))

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        if direction == RoadOption.CHANGELANELEFT or direction == RoadOption.CHANGELANERIGHT:
            steer, acc, brake = self._compute_function(image_input, speed, direction, self.lanechange_network)
        else:
            steer, acc, brake = self._compute_function(image_input, speed, direction, self.drive_network)

        '''
        if self._speed >= 25:
            acc = 0
        '''

        self.run_radar()
        self.traffic_light_detection()

        if self._obstacle_ahead or self.is_traffic_light_ahead():
            brake = 1.0
            acc = 0.0
        else:
            brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(acc)
        control.brake = float(brake)

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _compute_function(self, image_input, speed, control_input, network):

        branches = network.network_tensor
        x = network.input_images
        dout = network.dout
        input_speed = network.input_data[1]

        image_input = image_input.reshape(
            (1, network.image_size[0], network.image_size[1], network.image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed)

        speed = speed.reshape((1, 1))

        if control_input == RoadOption.LEFT:
            all_net = branches[0]
        elif control_input == RoadOption.RIGHT:
            all_net = branches[1]
        elif control_input == RoadOption.STRAIGHT:
            all_net = branches[2]
        elif control_input == RoadOption.LANEFOLLOW:
            all_net = branches[3]
        elif control_input == RoadOption.CHANGELANELEFT:
            all_net = branches[2]
        elif control_input == RoadOption.CHANGELANERIGHT:
            all_net = branches[3]
        else:
            all_net = branches[3]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(network.dropout_vec)}

        output_all = network.sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        return predicted_steers, predicted_acc, predicted_brake

    def get_high_level_command(self, convert=True):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
            from agents.navigation.local_planner import RoadOption
            if _hcl == RoadOption.LEFT:
                return 1
            elif _hcl == RoadOption.RIGHT:
                return 2
            elif _hcl == RoadOption.STRAIGHT:
                return 3
            elif _hcl == RoadOption.LANEFOLLOW:
                return 4
            elif _hcl == RoadOption.CHANGELANELEFT:
                return 5
            elif _hcl == RoadOption.CHANGELANERIGHT:
                return 6

        # return self._local_planner.get_high_level_command()
        hcl = self._local_planner.get_high_level_command()
        if convert:
            return hcl_converter(hcl)
        else:
            return self._local_planner.get_high_level_command()

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()

    def set_radar_data(self, radar_data):
        self._radar_data = radar_data

    def set_stop_radar_range(self):
        hcl = self._local_planner.get_high_level_command()
        c = self._vehicle.get_control()
        steer = abs(c.steer) * 15
        # 교차로 주행 시
        if hcl is RoadOption.RIGHT or hcl is RoadOption.LEFT or hcl is RoadOption.STRAIGHT:
            yaw_angle = 30
        else:  # 교차로 아닌 경우
            yaw_angle = 15
        return yaw_angle + steer

    def is_obstacle_ahead(self, _rotation, _detect):
        radar_range = self.set_stop_radar_range()

        threshold = max(self._speed * 0.2, 3)
        if -4.5 <= _rotation.pitch <= 5.0 and -radar_range <= _rotation.yaw <= radar_range and \
                _detect.depth <= threshold:
            return True
        return False

    def run_radar(self):
        if self._radar_data is None:
            return False
        current_rot = self._radar_data.transform.rotation
        self._obstacle_ahead = False
        if time.time() - self.traffic_light_detected_time > self.traffic_light_duration:
            self._is_traffic_light_in_distance = False

        for detect in self._radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)

            rotation = carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=azi,
                roll=current_rot.roll)

            if self.is_obstacle_ahead(rotation, detect):
                self._obstacle_ahead = True

            if -8 <= azi <= 8 and 2 <= alt <= 10 and 20 <= detect.depth <= 40:
                self._is_traffic_light_in_distance = True
                self.traffic_light_detected_time = time.time()

    def reset_destination(self):
        sp = self._map.get_spawn_points()
        rand_sp = random.choice(sp)
        self._vehicle.set_transform(rand_sp)

        control_reset = carla.VehicleControl()
        control_reset.steer, control_reset.throttle, control_reset.brake = 0.0, 0.0, 0.0
        self._vehicle.apply_control(control_reset)

        spawn_point = random.choice(self._map.get_spawn_points())
        self.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z),
                             start_loc=(rand_sp.location.x, rand_sp.location.y, rand_sp.location.z))

    def traffic_light_detection(self):
        if self.traffic_light_image is None:
            return

        image_cut = [0, 256, 272, 528]
        array = np.frombuffer(self.traffic_light_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.traffic_light_image.height, self.traffic_light_image.width, 4))

        frame_size = (self.traffic_image_input_size, self.traffic_image_input_size)
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]
        array = array[:, :, ::-1]

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        # 원하는 크기로 리사이즈
        image_pil = image_pil.resize((self.traffic_image_input_size, self.traffic_image_input_size))
        # image_pil.save('output/%06f.png' % time.time())
        np_image = np.array(image_pil, dtype=np.dtype("uint8"))

        # mask out side
        np_image[:, :int(self.traffic_image_input_size * 0.15)] = 0
        np_image[:, int(self.traffic_image_input_size * 0.85):self.traffic_image_input_size] = 0
        np_image[:int(self.traffic_image_input_size * 0.25)] = 0
        np_image[int(self.traffic_image_input_size * 0.65):] = 0

        image_raw = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preporcess(np.copy(image_raw),
                                            [self.traffic_image_input_size, self.traffic_image_input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.traffic_sess.run(
            [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
            feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.traffic_image_input_size,
                                         score_threshold=0.6)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        self.bounding_boxes = bboxes

    def is_traffic_light_ahead(self):
        if self.bounding_boxes is None:
            return False

        if len(self.bounding_boxes) > 0 and self._is_traffic_light_in_distance and self._is_stop_line_ahead():
            return True
        else:
            return False

    def _is_stop_line_ahead(self):
        if self.traffic_light_image is None:
            return
        slope = -12
        threshold = int(max(0, slope * self._speed + 350))
        image_cut = [threshold, 600, 320, 480]
        image_resize = (88, 200, 3)

        array = np.frombuffer(self.traffic_light_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.traffic_light_image.height, self.traffic_light_image.width, 4))
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]  # 필요 없는 부분을 잘라내고
        array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        array = cv2.GaussianBlur(array, (7, 7), 0)
        array = cv2.Canny(array, 80, 120)

        ret = np.any(array > 0)
        return ret
