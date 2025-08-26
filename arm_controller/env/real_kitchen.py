import gym
import panda_py
from gym import spaces
import numpy as np
import cv2
from traits.trait_types import self

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
        self._reset_robot_to_initial_state()
        obs = self._get_observation()
        return obs

    def step(self, action: int):
        raise NotImplementedError

    def render(self, mode='human'):
        image = self.camera.get_frame()
        return crop_and_resize(image, self.image_shape[0])

    def _get_observation(self) -> np.ndarray:
        q = self.arm.q
        return q

    def _replay_trajectory(self, file):
        full_path = f"{self.traj_dir}/{file}.h5"
        try:
            replay = Replay(self.arm)
            replay.replay_trajectory(path=full_path)
        except Exception as e:
            print(f"轨迹重播失败: {e}")

    def _reset_robot_to_initial_state(self):
        print("Resetting robot to initial position...")

        start_pos = [-0.000022460186500366954, -0.7836777044764737, 0.00041195781118179314, -2.3564412297671096,
                     -0.0009421655249574945, 1.5700842435025806, 0.7855311433151364]
        self._move_to(start_pos)

    def _release(self):
        width = 0.07983950525522232
        print(f"Grasping to width{width} speed{self.gripper_speed} force{self.gripper_force}")
        self.gripper.grasp(width, self.gripper_speed, self.gripper_force)

    def _grasp(self, width):
        print(f"Grasping to width{width} speed{self.gripper_speed} force{self.gripper_force}")
        self.gripper.grasp(width, self.gripper_speed, self.gripper_force)

    def _move_to(self, pos):
        self.arm.move_to_joint_position(np.array(pos))


class FridgeMangoCabJello(RealRobotSkillEnv):
    SKILL_LIBRARY = ['open_fridge', 'close_fridge', 'open_cab', 'store_mango', 'store_jello']

    def __init__(self):
        super().__init__()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

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
        else:
            raise NotImplementedError

        obs = self._get_observation()

        reward = 0.0
        done = False
        info = {"skill": self.SKILL_LIBRARY[action]}

        return obs, reward, done, info

    def _open_fridge(self):
        self._replay_trajectory('open_fridge_20250528160105')

    def _close_fridge(self):
        self._replay_trajectory('close_fridge_20250528160332')

    def _open_cab(self):
        self._replay_trajectory('open_cab_20250528155024')

    def _store_mango(self):
        # approach to mango
        pos = [0.17641643765735898, -0.24476337909280205, -0.1379896167013324, -2.7562025753824333,
               0.029356397736428428, 2.4453264166514073, 0.8679067428583899]
        self._move_to(pos)

        # go to mango grasping
        pos = [0.18807336746600656, 0.08113982689903493, -0.11537806052959966, -2.7579434441523825, 0.02804498908585973,
               2.80727768834432, 0.8052151337936521]
        self._move_to(pos)

        # grasp mango
        self._grasp(0.06522998213768005)

        # place mango
        self._replay_trajectory('place_mango_20250528160719')
        self._release()

        # back from fridge
        self._replay_trajectory('leave_fridge_20250528160810')

    def _store_jello(self):
        # approach to jello
        pos = [0.3845410173673113, -0.10737043349927154, 0.14711042294870225, -2.3757319257072838, 0.004609344418204302,
               2.0727093774482546, 0.5630765890752276]
        self._move_to(pos)

        # go to jello grasping
        pos = [0.3821798511488106, 0.22603936531080243, 0.14152096347570634, -2.507338585691338, 0.004422725360012716,
               2.7192444953215906, 1.309928688970705]
        self._move_to(pos)

        # grasp jello
        self._grasp(0.026925303041934967)

        # place jello
        self._replay_trajectory('place_jelly_20250528161109')
        self._release()

        # back from cab
        self._replay_trajectory('leave_cab_20250528161218')


class FruitsSnacks(RealRobotSkillEnv):
    SKILL_LIBRARY = ['open_fridge', 'close_fridge', 'open_cab', 'close_cab',
                     'store_mango', 'store_lemon', 'store_orange', 'store_cheezit', 'store_jello']

    def __init__(self):
        super().__init__()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 执行对应技能
        skill_fn = self.SKILL_LIBRARY[action]
        print(skill_fn)

        if action == 0:
            self._open_fridge()
        elif action == 1:
            self._close_fridge()
        elif action == 2:
            self._open_cab()
        elif action == 3:
            self._close_cab()
        elif action == 4:
            self._store_mango()
        elif action == 5:
            self._store_lemon()
        elif action == 6:
            self._store_orange()
        elif action == 7:
            self._store_cheezit()
        elif action == 8:
            self._store_jello()
        else:
            raise NotImplementedError

        obs = self._get_observation()

        reward = 0.0
        done = False
        info = {"skill": self.SKILL_LIBRARY[action]}

        return obs, reward, done, info

    def _open_fridge(self):
        self._replay_trajectory('open_fridge_720')
        self._reset2default_position()

    def _close_fridge(self):
        self._replay_trajectory('CLOSE_FRIDGE_20250704164844')
        self._reset2default_position()

    def _open_cab(self):
        self._replay_trajectory('OPEN_CAB_20250704171418')
        self._reset2default_position()

    def _close_cab(self):
        self._replay_trajectory('CLOSE_CAB_20250704171542')

    def _store_mango(self):
        # approach to mango
        pos = [0.17641643765735898, -0.24476337909280205, -0.1379896167013324, -2.7562025753824333,
               0.029356397736428428, 2.4453264166514073, 0.8679067428583899]
        self._move_to(pos)

        # go to mango grasping
        pos = [0.18807336746600656, 0.08113982689903493, -0.11537806052959966, -2.7579434441523825, 0.02804498908585973,
               2.80727768834432, 0.8052151337936521]
        self._move_to(pos)

        # grasp mango
        self._grasp(0.06522998213768005)

        # place mango
        self._replay_trajectory('PLACE_MANGO_20250704165218')
        self._release()

        # back from fridge
        self._replay_trajectory('LEAVE_FRIDGE_20250704165319')

        self._reset2default_position()

    def _store_lemon(self):
        # approach to lemon
        pos = [-0.03034798651358537, -0.35328322547561225, -0.28095724243448494, -2.800399724854488,
               -0.3620582289595172, 2.3662763739404036, 0.8007882299619037]
        self._move_to(pos)

        # go to lemon grasping
        pos = [-0.03038732587420153, -0.04969926388574523, -0.3646589120208171, -2.9009254678832184, -0.360250900104458,
               2.866468754382454, 0.8001443488595927]
        self._move_to(pos)

        # grasp lemon
        self._grasp(0.06427190452814102)

        # place lemon
        self._replay_trajectory('place_lemon_20250704182350')
        self._release()

        # back from fridge
        self._replay_trajectory('leave_lemon_20250704182350')

        self._reset2default_position()

    def _store_orange(self):
        # approach orange
        self._replay_trajectory('pick_orange_20250704181110')

        # grasp orange
        self._grasp(0.07438983023166656)

        # place orange
        self._replay_trajectory('place_orange_20250704181110')
        self._release()

        # back from fridge
        self._replay_trajectory('leave_orange_20250704181110')

        self._reset2default_position()

    def _store_cheezit(self):
        # approach to cheezit
        pos = [-0.4113092428001848, -0.5238304134217908, -0.3540462238704949, -2.6914061675482497, -0.27266361182813614,
               2.3656230286642748, -0.6034685147989678]
        self._move_to(pos)

        # go to cheezit grasping
        pos = [-0.4269500965005481, -0.15602347311884235, -0.377613943346855, -2.714064571915684, -0.2722122340621771,
               2.5734913837810005, -0.601038620368474]
        self._move_to(pos)

        # grasp cheezit
        self._grasp(0.04293417930603027)

        # place cheezit
        self._replay_trajectory('place_cheezit_20250704180357')
        self._release()
        self._replay_trajectory('push_cheezit_20250704180513')

        self._reset2default_position()

    def _store_jello(self):
        # approach to jello
        pos = [0.3845410173673113, -0.10737043349927154, 0.14711042294870225, -2.3757319257072838, 0.004609344418204302,
               2.0727093774482546, 0.5630765890752276]
        self._move_to(pos)

        # go to jello grasping
        pos = [0.3821798511488106, 0.22603936531080243, 0.14152096347570634, -2.507338585691338, 0.004422725360012716,
               2.7192444953215906, 1.309928688970705]
        self._move_to(pos)

        # grasp jello
        self._grasp(0.026925303041934967)

        # place jello
        self._replay_trajectory('PLACE_JELLO_20250704165726')
        self._release()

        # back from cab
        self._replay_trajectory('LEAVE_CAB_20250704165803')

        self._reset2default_position()

    def _reset2default_position(self):
        pos = [0.8024984602915509, -0.7170319747069731, -0.31783231969360565, -1.9078798450397978, -0.1717547503006375,
               1.3394796582349422, 0.9985520389668511]
        self._move_to(pos)


class HeatBread(RealRobotSkillEnv):
    SKILL_LIBRARY = ['open_microwave', 'close_microwave', 'set_time', 'move_bread_to_microwave',
                     'move_bread_to_plate']

    def __init__(self):
        super().__init__()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 执行对应技能
        skill_fn = self.SKILL_LIBRARY[action]
        print(skill_fn)

        if action == 0:
            self._open_microwave()
        elif action == 1:
            self._close_microwave()
        elif action == 2:
            self._set_time()
        elif action == 3:
            self._move_bread_to_microwave()
        elif action == 4:
            self._move_bread_to_plate()
        else:
            raise NotImplementedError

        obs = self._get_observation()

        reward = 0.0
        done = False
        info = {"skill": self.SKILL_LIBRARY[action]}

        return obs, reward, done, info

    def _open_microwave(self):
        self._replay_trajectory('微波炉按钮_821_v3')
        self._reset2default_position()

    def _close_microwave(self):
        self._replay_trajectory('关微波炉容错v3')
        self._reset2default_position()

    def _set_time(self):
        self._replay_trajectory('move_to_switch_821')
        self._grasp(0.005204739980399609)
        self._replay_trajectory('switch_821')
        self._release()

        self._reset2default_position()

    def _move_bread_to_microwave(self):
        self._replay_trajectory('move_to_bread_821_v2')
        self._grasp(0.026925303041934967)
        self._replay_trajectory('place_bread_821_v2')
        self._release()
        self._replay_trajectory('leave_mircowave_v2')

        self._reset2default_position()

    def _move_bread_to_plate(self):
        self._replay_trajectory('get_bread_from_micro_821')
        self._grasp(0.026925303041934967)
        self._replay_trajectory('place_bread_plate_821')
        self._release()

        self._reset2default_position()

    def _reset2default_position(self):
        pos = [0.8024984602915509, -0.7170319747069731, -0.31783231969360565, -1.9078798450397978, -0.1717547503006375,
               1.3394796582349422, 0.9985520389668511]
        self._move_to(pos)
