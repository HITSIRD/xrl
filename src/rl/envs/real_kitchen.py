import numpy as np
from collections import defaultdict
import gym
from gym import spaces
import numpy as np
from typing import Callable, Dict, Any

from src.utils.general import AttrDict, ParamDict
from src.rl.components.environment import GymEnv
import arm_controller.env


class RealKitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['mango', 'jelly', 'open fridge', 'close fridge']
    skill_library = {}

    for i, task in enumerate(SUBTASKS):
        skill_library[i] = task

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
