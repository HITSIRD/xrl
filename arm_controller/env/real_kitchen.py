import gym
import panda_py
from gym import spaces
import numpy as np
import cv2

from arm_controller.configs.config import ARM_URL
from arm_controller.src.controllers.camera import Camera
from arm_controller.src.replay.replay import Replay
from src.utils.process_dataset import crop_and_resize


class RealRobotSkillEnv(gym.Env):
    SKILL_LIBRARY = ['open_fridge', 'close_fridge', 'open_cab', 'store_mango', 'store_jello']

    def __init__(self, image_shape=(256, 256, 3)):
        super().__init__()
        self.image_shape = image_shape
        self.arm = panda_py.Panda(ARM_URL)
        self.gripper = panda_py.libfranka.Gripper(ARM_URL)
        self.camera = Camera()

        self.gripper_speed = 0.03
        self.gripper_force = 10.0

        self.traj_dir = '/home/user/文档/projects/xrl/arm_controller/data/traj'

        # Action 是离散的技能索引
        self.action_space = spaces.Discrete(len(self.SKILL_LIBRARY))

        # Observation 是图像：uint8 RGB 图像
        self.observation_space = spaces.Box(
            low=0, high=255, shape=image_shape, dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        # 可以加一个复位技能或复位操作
        self._reset_robot_to_initial_state()
        obs = self._get_observation()
        return obs

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 执行对应技能
        skill_fn = self.SKILL_LIBRARY[action]
        print(skill_fn)

        if action == 0:  # open fridge
            self._open_fridge()
        elif action == 1:
            self._close_fridge()
        elif action == 2:
            self._open_cab()
        elif action == 3:
            self._store_mango()
        elif action == 4:
            self._store_jello()

        # 获取新的观测
        obs = self._get_observation()

        # 这里暂时没有 reward 和 done 逻辑，你可以后续加上
        reward = 0.0
        done = False
        info = {"skill": self.SKILL_LIBRARY[action]}

        return obs, reward, done, info

    def render(self, mode='human'):
        # 可选：显示图像
        image = self.camera.get_frame()
        return crop_and_resize(image, self.image_shape[0])

    def _open_fridge(self):
        file_name = 'open_fridge_20250528160105'
        full_path = f"{self.traj_dir}/{file_name}.h5"

        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

    def _close_fridge(self):
        file_name = 'close_fridge_20250528160332'
        full_path = f"{self.traj_dir}/{file_name}.h5"

        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

    def _open_cab(self):
        file_name = 'open_cab_20250528155024'
        full_path = f"{self.traj_dir}/{file_name}.h5"
        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

    def _store_mango(self):
        # approach to mango
        pos = [0.17641643765735898, -0.24476337909280205, -0.1379896167013324, -2.7562025753824333,
               0.029356397736428428, 2.4453264166514073, 0.8679067428583899]
        self.arm.move_to_joint_position(np.array(pos))

        # go to mango grasping
        pos = [0.18807336746600656, 0.08113982689903493, -0.11537806052959966, -2.7579434441523825, 0.02804498908585973,
               2.80727768834432, 0.8052151337936521]
        self.arm.move_to_joint_position(np.array(pos))

        # grasp mango
        width = 0.06522998213768005
        print(f"Grasping to width{width} speed{self.gripper_speed} force{self.gripper_force}")
        self.gripper.grasp(width, self.gripper_speed, self.gripper_force)

        # place mango
        file_name = 'place_mango_20250528160719'
        full_path = f"{self.traj_dir}/{file_name}.h5"
        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

        # release
        width = 0.07983950525522232
        print(f"Grasping to width{width} speed{self.gripper_speed} force{self.gripper_force}")
        self.gripper.grasp(width, self.gripper_speed, self.gripper_force)

        # back from fridge
        file_name = 'leave_fridge_20250528160810'
        full_path = f"{self.traj_dir}/{file_name}.h5"
        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

    def _store_jello(self):
        # approach to jello
        pos = [0.3845410173673113, -0.10737043349927154, 0.14711042294870225, -2.3757319257072838, 0.004609344418204302,
               2.0727093774482546, 0.5630765890752276]
        self.arm.move_to_joint_position(np.array(pos))

        # go to jello grasping
        pos = [0.3821798511488106, 0.22603936531080243, 0.14152096347570634, -2.507338585691338, 0.004422725360012716,
               2.7192444953215906, 1.309928688970705]
        self.arm.move_to_joint_position(np.array(pos))

        # grasp jello
        width = 0.026925303041934967
        print(f"Grasping to width{width} speed{self.gripper_speed} force{self.gripper_force}")
        self.gripper.grasp(width, self.gripper_speed, self.gripper_force)

        # place jello
        file_name = 'place_jelly_20250528161109'
        full_path = f"{self.traj_dir}/{file_name}.h5"
        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

        # release
        width = 0.07983950525522232
        print(f"Grasping to width{width} speed{self.gripper_speed} force{self.gripper_force}")
        self.gripper.grasp(width, self.gripper_speed, self.gripper_force)

        # back from cab
        file_name = 'leave_cab_20250528161218'
        full_path = f"{self.traj_dir}/{file_name}.h5"
        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

    def _get_observation(self) -> np.ndarray:
        q = self.arm.q
        return q

    def _reset_robot_to_initial_state(self):
        print("Resetting robot to initial position...")

        start_pos = [-0.000022460186500366954, -0.7836777044764737, 0.00041195781118179314, -2.3564412297671096,
                     -0.0009421655249574945, 1.5700842435025806, 0.7855311433151364]
        self.arm.move_to_joint_position(np.array(start_pos))
