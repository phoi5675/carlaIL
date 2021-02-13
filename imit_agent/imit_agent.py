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


class ImitAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """
    front_image = None

    def __init__(self, vehicle, avoid_stopping, target_speed=20, memory_fraction=0.25, image_cut=[115, 510]):
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
        self._hop_resolution = 1.5
        self._path_seperation_hop = 3
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

        # data from vehicle
        self.vehicle_speed = 0

        # load tf network model
        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        config_gpu = tf.ConfigProto()  # tf 설정 프로토콜인듯?

        # GPU to be selected, just take zero , select GPU  with CUDA_VISIBLE_DEVICES

        config_gpu.gpu_options.visible_device_list = '0'  # GPU >= 2 일 때, 첫 번째 GPU만 사용

        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction  # memory_fraction % 만큼만 gpu vram 사용

        self._image_size = (88, 200, 3)  # 아마 [세로, 가로, 차원(RGB)] 인듯?
        self._avoid_stopping = avoid_stopping

        self._sess = tf.Session(config=config_gpu)  # 작업을 위한 session 선언

        with tf.device('/gpu:0'):  # 수동으로 device 배치 / default : '/gpu:0'
            # tf.placeholder(dtype, shape, name) 형태로 shape에 데이터를 parameter로 전달
            self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                self._image_size[1],
                                                                self._image_size[2]],
                                                name="input_image")

            self._input_data = []

            # input control 종류가 4가지니까 [None, 4]로 지정?
            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 4], name="input_control"))

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

        direction = self.get_high_level_command()
        v = self._vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)  # use m/s

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

        rgb_image.convert(cc.Raw)

        array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
        array = array[self._image_cut[0]:self._image_cut[1], :, :3]  # 필요 없는 부분을 잘라내고
        # array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        image_pil = image_pil.resize((self._image_size[1], self._image_size[0]))  # 원하는 크기로 리사이즈
        image_input = np.array(image_pil, dtype=np.dtype("uint8"))

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        # Control() 대신 VehicleControl() 으로 변경됨 (0.9.X 이상)
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
        speed = np.array(speed / 1.0)

        speed = speed.reshape((1, 1))

        # index 의 의미를 알아야 네트워크 분석이 가능할듯..!
        '''
        control_input (directions) 
        REACH_GOAL = 0.0
        GO_STRAIGHT = 5.0
        TURN_RIGHT = 4.0
        TURN_LEFT = 3.0
        LANE_FOLLOW = 2.0
        '''
        '''
        if control_input == 2 or control_input == 0.0:
            all_net = branches[0]   # continue
        elif control_input == 3:
            all_net = branches[2]   # left
        elif control_input == 4:
            all_net = branches[3]   # right
        elif control_input == 5:
            all_net = branches[1]   # straight?
        '''
        if control_input == 2 or control_input == 0.0:
            all_net = branches[0]  # continue
        elif control_input == 3:
            all_net = branches[1]  # left
        elif control_input == 4:
            all_net = branches[2]  # right
        elif control_input == 5:
            all_net = branches[3]  # straight?

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        if self._avoid_stopping:
            predicted_speed = sess.run(branches[4], feed_dict=feedDict)
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

    def get_high_level_command(self):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
            from agents.navigation.local_planner import RoadOption
            REACH_GOAL = 0.0
            GO_STRAIGHT = 5.0
            TURN_RIGHT = 4.0
            TURN_LEFT = 3.0
            LANE_FOLLOW = 2.0

            if _hcl == RoadOption.STRAIGHT:
                return GO_STRAIGHT
            elif _hcl == RoadOption.LEFT:
                return TURN_LEFT
            elif _hcl == RoadOption.RIGHT:
                return TURN_RIGHT
            elif _hcl == RoadOption.LANEFOLLOW or _hcl == RoadOption.VOID:
                return LANE_FOLLOW
            else:
                return REACH_GOAL

        hcl = self._local_planner.get_high_level_command()
        return hcl_converter(hcl)

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()
