import numpy as np
import d4rl

from strl.rl.components.environment import GymEnv
from strl.utils.general_utils import ParamDict, AttrDict


class MazeEnv(GymEnv):
    """Shallow wrapper around gym env for maze envs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _default_hparams(self):
        default_dict = ParamDict({
        })
        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        super().reset()
        if self.TARGET_POS is not None and self.START_POS is not None:
            noise = np.clip(np.random.normal(0, 1.0), -1, 1)
            self._env.set_target(self.TARGET_POS + noise)
            self._env.reset_to_location(self.START_POS)
        self._env.render(mode='rgb_array')  # these are necessary to make sure new state is rendered on first frame
        obs, _, _, _ = self._env.step(np.zeros_like(self._env.action_space.sample()))
        return self._wrap_observation(obs)

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        # self._env.render(mode='human')
        if rew > 0.5:
            done = True
        return obs, np.float64(rew), done, info  # casting reward to float64 is important for getting shape later


class ACRandMaze0S40Env(MazeEnv):
    START_POS = np.array([10., 24.])
    TARGET_POS = np.array([18., 8.])

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-randMaze0S40-ac-v0",
        })
        return super()._default_hparams().overwrite(default_dict)


class ACRandMaze0S30Env(MazeEnv):
    START_POS = np.array([10., 24.])
    TARGET_POS = np.array([18., 6.])

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-randMaze0S30-ac-v0",
        })
        return super()._default_hparams().overwrite(default_dict)


class ACRandMaze0S20Env(MazeEnv):
    START_POS = np.array([6.0, 12.0])
    TARGET_POS = np.array([19.0, 10.0])

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-randMaze0S20-ac-v0",
        })
        return super()._default_hparams().overwrite(default_dict)
