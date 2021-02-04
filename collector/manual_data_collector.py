#!/usr/bin/env python

import carla
from game import *
from Recorder import *
import argparse
import pygame

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    tick_double_time = False  # delta_seconds 의 두 배 시간으로 녹화하게 만듦
    try:
        # 서버 연결
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        # pygame 해상도 설정 / HWSURFACE, DOUBLEBUF 는 화면 플리커링 등 방지용
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        # TODO 본 테스트 할 때는 sync 모드 켜기
        # set synchronous mode
        server_world = client.get_world()
        settings = server_world.get_settings()
        # settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05

        server_world.apply_settings(settings)

        world = World(server_world, hud, args)

        controller = KeyboardControl(world, args.autopilot)
        game_controller = JoyStickControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            # tick_busy_loop(FPS) : 수직동기화랑 비슷한 tick() 함수
            clock.tick_busy_loop(30)
            game_controller.parse_events(client, world, clock)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            # server_world.tick()  # 서버 시간 tick

            # 화면 제외 다른 데이터 녹화
            if world.recording_enabled and tick_double_time:
                Recorder.record(world, path=args.path)
                tick_double_time = False
            elif tick_double_time is False:  # 스위치 기능을 이용해 time_step 을 0.05초의 두 배로 설정
                tick_double_time = True

            world.render(display)
            pygame.display.flip()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--path',
        default='output/',
        help='path for saving data')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

