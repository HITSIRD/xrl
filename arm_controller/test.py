"""
Simple teaching demonstration. Teaches three joint positions
and connects them using motion generators. Also teaches a joint
trajectory that is then replayed.
"""
import os
import sys
import time

import h5py
import panda_py
import numpy as np
from panda_py import libfranka
# import matplotlib.pyplot as plt
import cv2

from configs.config import ARM_URL
from arm_controller.env.real_kitchen import RealRobotSkillEnv
from arm_controller.src.controllers.hand_controller import HandController
from src.logger.traj_logger import TrajectoryLogger
from src.replay.replay import Replay
from src.sampler.sampler import Sampler


def test_gripper():
    gripper = libfranka.Gripper(ARM_URL)
    gripper.move(0.08, 0.03)
    gripper.grasp(0.03, 0.03, 10)
    # gripper.move(0, 0.03)
    # output = gripper.read_once()
    # print(output.time)
    # print(output.width)
    # print(output.is_grasped)
    # print(output.max_width)
    # print(output.temperature)


def test_hand_controller():
    hand = libfranka.Gripper(ARM_URL)
    hand_controller = HandController(hand)
    hand_controller.release_gripper()
    print(f"read width: {hand_controller.read_width()}")
    # hand_controller.grasp(0.03, 0.03, 10)
    # print(f"read width: {hand_controller.read_width()}")


def test_arm_sampler():
    arm = panda_py.Panda(ARM_URL)
    print(f"j_goto:{arm.get_state().q}")

    print("position:", arm.get_position())
    print("orientation:", arm.get_orientation())


def test_npy():
    data = np.load(
        '/home/user/文档/projects/robot_arm_django/home/arm_controller/data/collection/20250521053402/camera.npy')
    print("Data shape:", data.shape)


def test_h5():
    with h5py.File(
            '/home/user/文档/projects/robot_arm_django/home/arm_controller/data/collection/fridge_mango_cab_jelly_7/traj.h5',
            'r') as f:
        print(f"Keys in HDF5 file: {list(f.keys())}")
        last_skill = f['skill'][0]
        print(last_skill.decode('utf-8'))
        for i in range(len(f['skill'])):
            current_skill = f['skill'][i]
            if current_skill != last_skill:
                print(current_skill.decode('utf-8'), i)
                last_skill = f['skill'][i]

        plt.imsave(f'obs_{0}.png', f['rgb'][0])
        plt.imsave(f'obs_{100}.png', f['rgb'][100])
        plt.imsave(f'obs_{1000}.png', f['rgb'][1000])
        # print(f['skill'][100].decode('utf-8'))
        # print(f['skill'][200].decode('utf-8'))
        # print(f['skill'][300].decode('utf-8'))
        # print(f['pose'][0])
        # print(f['pose'][1])
        # print(f['pose'][100])
        # print(f['pose'][200])
        # print(f['pose'][201])
        # print(f['pose'][300])
        # for i in range(len(f['skill'])):
        #     print(f['skill'][i].decode('utf-8'))
        #     print(f['pose'][i])
        #     plt.imsave(f'obs_{i}.png', f['rgb'][i])


def replay_trajectory(path, arm):
    """
    重播指定的轨迹文件。

    参数:
    traj_name (str): HDF5 文件名称（不含路径），例如 "example_trajectory.h5"。
    path (str): 文件路径。
    playback_frequency (int): 回放频率（默认为 1000 Hz）。
    """
    h5_path = path

    # 加载 HDF5 文件中的 q 和 dq 数据
    with h5py.File(h5_path, 'r') as h5f:
        q = h5f['q'][:]

    print(f"Loaded trajectory data from {h5_path}.")
    print(f"Trajectory length: {len(q)} steps.")

    # # 提示按回车键开始重播
    # input("Press Enter to replay trajectory.")

    # 将机械臂移动到初始位置（q[0]）
    arm.move_to_joint_position(q[0])
    print("Moved to initial joint position.")
    trajectory_length = len(q)
    for i in range(trajectory_length):
        arm.move_to_joint_position(q[i])

    print("Replay completed.")


def camera_test():
    import pyrealsense2 as rs
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        print("Warming up camera...")

        # 丢弃前面几帧
        for _ in range(30):
            pipeline.wait_for_frames()

        while True:
            # 连续取帧，确保每次取的是最新的
            for _ in range(3):  # 丢弃2帧，取第3帧
                frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                print("No color frame!")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow("Color", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("[ERROR]", e)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def test_env():
    skill_library = {0: '', 1: '', 2: '', 3: '', 4: ''}
    env = RealRobotSkillEnv(skill_library)
    """ """
    env.step(0)


if __name__ == '__main__':
    arm = panda_py.Panda(ARM_URL)
    # replay_trajectory('/home/user/文档/projects/robot_arm_django/home/arm_controller/data/collection/test/traj.h5', arm)
    arm.move_to_start()
    # arm.recover()
    # print(arm.raise_error())
    # arm.move_to_start()
    # gripper = libfranka.Gripper(ARM_URL)
    # gripper.move(0.1, 0.03)
    # gripper.grasp(0.03, 0.03, 10)
    # sample = Sampler(arm)
    # print(f"current pos: {sample.sample_pos()}")
    # data = sample.sample_trajectory()
    # traj_logger = TrajectoryLogger()
    # traj_logger.write(data, "data/traj", "trajectory")

    # replay = Replay(arm)
    # replay.replay_trajectory(path='data/traj/trajectory.h5')
    # test_hand_controller()
    # test_arm_sampler()

    # test_h5()
    # camera_test()

    # test_env()
