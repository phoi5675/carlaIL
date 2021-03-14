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
import scipy
import os
import sys
import math
import cv2

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
from imitation_learning_network import load_imitation_learning_network
from agents.navigation.local_planner import RoadOption


class ImitAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """
    front_image = None

    def __init__(self, vehicle, avoid_stopping, target_speed=20, memory_fraction=0.25, image_cut=[220, 600]):
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
        self._hop_resolution = 0.2
        self._path_seperation_hop = 3
        self._path_seperation_threshold = 1.0
        self._target_speed = target_speed
        self._grp = None

        # data from vehicle
        self._speed = 0
        self._radar_data = None
        self._obstacle_ahead = False

        # load tf network model
        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 7

        config_gpu = tf.ConfigProto()  # tf 설정 프로토콜인듯?

        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

        config_gpu.gpu_options.visible_device_list = '0'  # GPU >= 2 일 때, 첫 번째 GPU만 사용

        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction  # memory_fraction % 만큼만 gpu vram 사용

        self._image_size = (88, 200, 3)  # 아마 [세로, 가로, 차원(RGB)] 인듯?
        self._avoid_stopping = avoid_stopping

        self._sess = tf.Session(config=config_gpu)  # 작업을 위한 session 선언

        with tf.device('/cpu:0'):  # 수동으로 device 배치 / default : '/gpu:0'
            # tf.placeholder(dtype, shape, name) 형태로 shape에 데이터를 parameter로 전달
            self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                self._image_size[1],
                                                                self._image_size[2]],
                                                name="input_image")

            self._input_data = []

            # input control 종류가 4가지니까 [None, 4]로 지정?
            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 6], name="input_control"))

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 1], name="input_speed"))

            # dropout vector 값. 아마 신경망이랑 관련 있는듯
            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):  # 아래의 network_tensor 가 Network 아래의 신경망임을 명시 -> Network/network_tensor 의 형태
            # 여기에 있는 load_imitation_learning_network 가 신경망 자체
            self._network_tensor = load_imitation_learning_network(self._input_images,
                                                                   self._input_data,
                                                                   self._image_size, self._dout)

        import os
        dir_path = os.path.dirname(__file__)

        self._models_path = dir_path + '/model/'

        # 그래프 초기화
        # tf.reset_default_graph()

        # 변수 초기화 -> 작업 전 명시적으로 수행 / session 실행
        self._sess.run(tf.global_variables_initializer())

        self.load_model()

        self._image_cut = image_cut

    def load_model(self):
        # 이전에 학습한 결과 로드
        variables_to_restore = tf.global_variables()

        # 모델, 파라미터 저장
        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        # checkpoint 가 존재하는 경우 로드
        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

        self._local_planner.change_intersection_hcl()

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

        control = self._compute_action(ImitAgent.front_image, speed, direction)
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

        image_cut = [230, 480, 160, 640]
        image_resize = (88, 200, 3)
        w = image_resize[1]
        h = image_resize[0]

        src = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
        dst = np.float32([[90, h], [110, h], [0, 0], [w, 0]])
        M = cv2.getPerspectiveTransform(src, dst)

        # carla.Image 를 기존 manual_control.py.CameraManager._parse_image() 부분을 응용
        rgb_image.convert(cc.Raw)
        array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]  # 필요 없는 부분을 잘라내고
        array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        image_pil = image_pil.resize((image_resize[1], image_resize[0]))  # 원하는 크기로 리사이즈
        # image_pil.save('output/%06d.png' % image.frame)
        np_image = np.array(image_pil, dtype=np.dtype("uint8"))

        # bird-eye view transform
        # https://nikolasent.github.io/opencv/2017/05/07/Bird%27s-Eye-View-Transformation.html
        np_image = cv2.warpPerspective(np_image, M, (w, h))

        np_image = cv2.Canny(np_image, 20, 60)

        # masking to extract region of interest(ROI)
        pts = np.array([[0, 0], [0, h], [90, h], [80, 45],
                        [120, 45], [110, h], [w, h], [w, 0]], dtype=np.int32)
        cv2.fillPoly(np_image, [pts], (0, 0, 0))

        # grayscale to rgb
        image_input = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        image = Image.fromarray(np_image)
        import random
        # image.save('output/%.3f.png' % random.uniform(0, 1000))

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        self.run_radar()
        if self._obstacle_ahead:
            brake = 1.0
        else:
            brake = 0.0
        '''
        # We limit speed to 35 km/h to avoid
        if speed > 25.0 and brake == 0.0:
            acc = 0.0
        '''
        '''
        # remove steering noise
        if steer <= -1:
            steer = -1
        elif steer >= 1:
            steer = 1

        if direction is RoadOption.LEFT and steer >= 0.3:
            steer = 0.3
        elif direction is RoadOption.RIGHT and steer <= -0.3:
            steer = -0.3
        elif direction is RoadOption.STRAIGHT or direction is RoadOption.LANEFOLLOW:
            if steer <= -0.4:
                steer = 0.4
            elif steer >= 0.4:
                steer = 0.4
        '''
        
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(acc)
        control.brake = float(brake)

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input, sess):

        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]

        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed * 1.0)

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
            all_net = branches[4]
        elif control_input == RoadOption.CHANGELANERIGHT:
            all_net = branches[5]
        else:
            all_net = branches[3]
            print("else")

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        if self._avoid_stopping:
            predicted_speed = sess.run(branches[6], feed_dict=feedDict)
            predicted_speed = predicted_speed[0][0]
            real_speed = speed * 25.0

            real_predicted = predicted_speed * 25.0
            if real_speed < 2.0 and real_predicted > 3.0:
                # If (Car Stooped) and
                #  ( It should not have stopped, use the speed prediction branch for that)

                predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc[0][0]

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

        for detect in self._radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)

            rotation = carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=azi,
                roll=current_rot.roll)

            if self.is_obstacle_ahead(rotation, detect):
                self._obstacle_ahead = True
