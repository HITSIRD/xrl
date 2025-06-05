import numpy as np
from collections import defaultdict
import gym
from gym import spaces
import numpy as np
from typing import Callable, Dict, Any

from src.utils.general import AttrDict, ParamDict
from src.rl.components.environment import GymEnv


class RealRobotSkillEnv(gym.Env):
    def __init__(self, skill_library: Dict[int, Callable[[], None]], image_shape=(128, 128, 3)):
        super().__init__()
        self.skill_library = skill_library
        self.image_shape = image_shape

        # Action 是离散的技能索引
        self.action_space = spaces.Discrete(len(skill_library))

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
        skill_fn = self.skill_library[action]
        skill_fn()

        # 获取新的观测
        obs = self._get_observation()

        # 这里暂时没有 reward 和 done 逻辑，你可以后续加上
        reward = 0.0
        done = False
        info = {"skill": action}

        return obs, reward, done, info

    def render(self, mode='human'):
        # 可选：显示图像
        image = self._get_observation()
        import cv2
        cv2.imshow("robot_cam", image)
        cv2.waitKey(1)

    def _get_observation(self) -> np.ndarray:
        """
        从相机获取RGB图像。
        请确保图像是np.uint8格式，并符合image_shape。
        """
        image = get_rgb_image()  # 你需要提供这个函数
        if image.shape != self.image_shape:
            image = cv2.resize(image, self.image_shape[:2][::-1])  # Resize to match shape
        return image

    def _reset_robot_to_initial_state(self):
        print("Resetting robot to initial position...")
        # 可以定义 skill 0 为 reset
        if 0 in self.skill_library:
            self.skill_library[0]()


class RealKitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['mango', 'jelly', 'open fridge', 'close fridge']
    skill_library = {}

    for i, task in enumerate(SUBTASKS):
        skill_library[i] = task

    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "kitchen-kbts-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        # return obs, np.float64(rew), done, self._postprocess_info(info)    # casting reward to float64 is important for getting shape later
        return obs, np.float64(rew), done, info  # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info.pop("completed_tasks")
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
        return info
