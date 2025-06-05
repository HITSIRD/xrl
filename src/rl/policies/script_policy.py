import torch
import numpy as np

from src.utils.general import AttrDict
from src.rl.components.agent import BaseAgent
from src.rl.components.policy import Policy


class ScriptPolicy(Policy):
    def __init__(self, config):
        self._hp = config
        # self.update_model_params(self._hp.prior_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None
        self.skill_enc = np.eye(5)

    def forward(self, obs, index=None):
        assert index is not None
        return AttrDict(action=index, action_index=index)

    def _build_network(self):
        return None

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

    @property
    def has_trainable_params(self):
        """Indicates whether policy has trainable params."""
        return False
