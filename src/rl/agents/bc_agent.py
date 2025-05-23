import torch
import os
import numpy as np
from torchviz import make_dot

from src.rl.components.agent import BaseAgent
from src.utils.general import ParamDict, map_dict, AttrDict
from src.utils.pytorch import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
# from src.rl.utils.mpi import sync_networks


class BCAgent(BaseAgent):
    """Implements actor-critic agent. (does not implement update function, this should be handled by RL algo agent)"""

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = config
        self._hp.policy_params.device = config.device
        self.policy = self._hp.policy(self._hp.policy_params)
        if self.policy.has_trainable_params:
            if hasattr(self._hp.policy_params, 'freeze_img_encoder'):
                if self._hp.policy_params.freeze_img_encoder:
                    self.policy.net.enc_p._image_enc.eval()
                    for p in self.policy.net.enc_p._image_enc.parameters():
                        p.requires_grad = False
            self.policy_opt = self._get_optimizer(self._hp.optimizer, self.policy, self._hp.policy_lr)

    def _act(self, obs, index=None):
        # TODO implement non-sampling validation mode
        obs = map2torch(obs, self._hp.device)
        if index is not None:
            return self.policy(obs, index)

        if isinstance(obs, AttrDict):
            return map2np(self.policy(obs))
        elif len(obs.shape) == 1:  # we need batched inputs for policy
            policy_output = self._remove_batch(self.policy(obs[None]))
            if 'dist' in policy_output:
                del policy_output['dist']
            return map2np(policy_output)
        return map2np(self.policy(obs))

    def _act_rand(self, obs):
        policy_output = self.policy.sample_rand(map2torch(obs, self.policy.device))
        if 'dist' in policy_output:
            del policy_output['dist']
        return map2np(policy_output)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        if self.policy.has_trainable_params:
            d['policy_opt'] = self.policy_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.policy_opt.load_state_dict(state_dict.pop('policy_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def visualize(self, logger, rollout_storage, step):
        super().visualize(logger, rollout_storage, step)
        self.policy.visualize(logger, rollout_storage, step)

    def reset(self):
        self.policy.reset()

    # def sync_networks(self):
    #     if self.policy.has_trainable_params:
    #         sync_networks(self.policy)

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        return experience_batch
