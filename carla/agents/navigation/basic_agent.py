#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption

import math
import random
import time


class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)

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
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

        self._radar_data = None
        self._obstacle_ahead = False
        self._obstacle_far_ahead = False
        self._speed = 0.0

        self.noise_steer_path = 0
        self.noise_steer = 0
        self.noise_start_time = 0
        self.noise_duration = 0

        self.weird_steer_count = 0
        self.weird_reset_count = 0

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

        self._local_planner.change_intersection_hcl()

        self.weird_steer_count = 0
        self.weird_reset_count = 0

        print("set new waypoint")

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

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        v = self._vehicle.get_velocity()
        c = self._vehicle.get_control()

        speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self._speed = speed

        self.run_radar()

        # check maneuvering
        if self.is_maneuvering_weird(c):
            self.weird_steer_count += 1

        if self.weird_steer_count >= 15:
            print("vehicle is steering in wrong way")
            self.weird_steer_count = 0
            self.weird_reset_count += 1

        if self.weird_reset_count >= 3:
            self.reset_destination()
            self.weird_reset_count = 0
            self.weird_steer_count = 0

        control = self._local_planner.run_step(debug=debug)

        self._state = AgentState.NAVIGATING
        # standard local planner behavior

        if self._obstacle_ahead:
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
        # control.throttle = self._vehicle_throttle

        return control

    # ====================================================================
    # ----- appended from original code ----------------------------------
    # ====================================================================
    def get_high_level_command(self):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
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
            '''
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
            '''

        # return self._local_planner.get_high_level_command()
        hcl = self._local_planner.get_high_level_command()
        return hcl_converter(hcl)

    def is_maneuvering_weird(self, control):
        hcl = self._local_planner.get_high_level_command()
        turn_threshold = 0.7
        if hcl is RoadOption.STRAIGHT or hcl is RoadOption.LANEFOLLOW:
            if abs(control.steer) >= turn_threshold:
                return True
        elif (hcl is RoadOption.LEFT or hcl is RoadOption.CHANGELANELEFT) and control.steer >= turn_threshold:
            return True
        elif (hcl is RoadOption.RIGHT or hcl is RoadOption.CHANGELANERIGHT) and control.steer <= -turn_threshold:
            return True
        elif (hcl is RoadOption.CHANGELANERIGHT or hcl is RoadOption.CHANGELANELEFT) \
                and abs(control.steer) >= turn_threshold:
            return True
        else:
            return False

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()

    def is_dest_far_enough(self):
        return self._local_planner.is_dest_far_enough()

    def set_radar_data(self, radar_data):
        self._radar_data = radar_data

    def set_stop_radar_range(self):
        hcl = self._local_planner.get_high_level_command()
        c = self._vehicle.get_control()
        steer = abs(c.steer) * 20
        # 교차로 주행 시
        if hcl is RoadOption.RIGHT or hcl is RoadOption.LEFT or hcl is RoadOption.STRAIGHT:
            yaw_angle = 40
        else:  # 교차로 아닌 경우
            yaw_angle = 15
        return yaw_angle + steer

    def set_target_speed(self, speed):
        self._local_planner.set_target_speed(speed)

    def is_obstacle_ahead(self, _rotation, _detect):
        radar_range = self.set_stop_radar_range()

        threshold = max(self._speed * 0.2, 3)
        if -5.0 <= _rotation.pitch <= 5.0 and -radar_range <= _rotation.yaw <= radar_range and \
                _detect.depth <= threshold:
            return True
        return False

    def is_obstacle_far_ahead(self, _rotation, _detect):
        radar_range = self.set_stop_radar_range()
        c = self._vehicle.get_control()
        steer = abs(c.steer) * 4
        left_margin = steer if c.steer < 0 else 0
        right_margin = steer if c.steer > 0 else 0
        if 1.0 <= _rotation.pitch <= 5.0 and -(5 + left_margin) <= _rotation.yaw <= (5 + right_margin) \
                and _detect.depth <= 15:
            return True
        return False

    def run_radar(self):
        if self._radar_data is None:
            return False
        current_rot = self._radar_data.transform.rotation
        self._obstacle_ahead = False
        self._obstacle_far_ahead = False

        for detect in self._radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)

            rotation = carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=azi,
                roll=current_rot.roll)

            if self.is_obstacle_ahead(rotation, detect):
                self._obstacle_ahead = True
            elif self.is_obstacle_far_ahead(rotation, detect):
                self._obstacle_far_ahead = True

        '''
        if self._obstacle_far_ahead:
            target_spd = 17
        else:
            target_spd = 28
        
        self._local_planner.set_target_speed(target_spd)
        self._target_speed = target_spd
        '''

    def noisy_agent(self, control):
        """

        :return: steer_noise
        """
        noisy_time = time.time() - self.noise_start_time
        if noisy_time >= self.noise_duration:
            self.noise_steer = random.choice([random.uniform(0.2, 0.4), random.uniform(-0.4, -0.2)])
            self.noise_start_time = time.time()
            self.noise_duration = random.uniform(2, 5)
        elif self.noise_duration * 0.5 < noisy_time <= self.noise_duration:
            return 0  # 노이즈 기간의 남은 30% 동안은 노이즈 삽입 X

        return self.noise_steer

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
