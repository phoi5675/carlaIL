import math
from PIL import Image


class Recorder(object):
    # High-level commands
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    CHANGELEFT = 4
    CHANGERIGHT = 5
    LANEFOLLOW = 6
    TRLIGHT_STOP = 7  # 이거 제외하고 나머지는 원래 있던 변수

    # 순서는 carla IL 의 datasheet 순서를 따름
    image = None

    steer = 0.0
    gas = 0.0
    brake = 0.0
    hand_brake = False
    reverse_gear = False
    steer_noise = 0.0  # noise 값은 학습에 영향 x
    gas_noise = 0.0
    brake_noise = 0.0
    x_pos = 0.0
    y_pos = 0.0
    speed = 0.0
    # 센서에서 얻는 데이터
    collision_other = 0.0
    collision_ped = 0.0
    collision_car = 0.0
    opposite_line_enter = 0.0
    sidewalk_intersect = 0.0
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0
    platform_time = 0.0
    game_time = 0.0  # not used
    orientation_x = 0.0
    orientation_y = 0.0
    orientation_z = 0.0

    high_level_com = 0
    noise = False  # not used
    camera = 0  # not used
    angle = 0  # not used

    @staticmethod
    def record(world):
        DATA_ARY = [Recorder.steer, Recorder.gas, Recorder.brake, Recorder.hand_brake, Recorder.reverse_gear,
                    Recorder.steer_noise, Recorder.gas_noise, Recorder.brake_noise,
                    Recorder.x_pos, Recorder.y_pos, Recorder.speed,
                    Recorder.collision_other, Recorder.collision_ped, Recorder.collision_car,
                    Recorder.opposite_line_enter, Recorder.sidewalk_intersect,
                    Recorder.acc_x, Recorder.acc_y, Recorder.acc_z,
                    Recorder.platform_time, Recorder.game_time,
                    Recorder.orientation_x, Recorder.orientation_y, Recorder.orientation_z,
                    Recorder.high_level_com, Recorder.noise, Recorder.camera, Recorder.angle]

        v = world.player.get_velocity()
        t = world.player.get_transform()
        c = world.player.get_control()
        acc = world.player.get_acceleration()

        # 순서는 carla IL 의 datasheet 순서를 따름
        Recorder.steer = c.steer
        Recorder.gas = c.throttle
        Recorder.brake = c.brake
        Recorder.hand_brake = c.hand_brake
        Recorder.reverse_gear = c.reverse

        Recorder.x_pos = t.location.x
        Recorder.y_pos = t.location.y
        Recorder.speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        Recorder.acc_x = acc.x
        Recorder.acc_y = acc.y
        Recorder.acc_z = acc.z
        Recorder.platform_time = world.hud.simulation_time
        # Recorder.game_time
        Recorder.orientation_x = t.rotation.roll
        Recorder.orientation_y = t.rotation.pitch
        Recorder.orientation_z = t.rotation.yaw

        for _list in DATA_ARY:
            print(_list, end=' ')
        print('\n')
        if Recorder.image is not None:  # 동기화 문제 때문에 image 가 None 상태 유지되는 경우 예방
            Recorder.image.save('output/%.3f.png' % Recorder.platform_time)
