import torch
import numpy as np

from src.rl.components.agent import BaseAgent
from src.utils.pytorch import ten2ar, map2torch, map2np
from src.utils.general import ParamDict, map_dict, AttrDict


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = config
        self.policy = self._hp.policy(self._hp.policy_params)

        # build replay buffer
        self.replay_buffer = self._hp.replay(self._hp.replay_params)
        self.policy_opt = self._get_optimizer(self._hp.optimizer, self.policy.q_eval, self._hp.policy_lr)
        self._update_steps = 0

        # if not self._hp.update_codebook:
        #     self.policy.codebook.requires_grad = False

    def update(self, experience_batch):
        self.add_experience(experience_batch)

        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self._sample_experience()
            experience_batch = self._normalize_batch(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._preprocess_experience(experience_batch)

            # policy_output = self._run_policy(experience_batch.observation)

            batch_idx = torch.arange(self._hp.batch_size, dtype=torch.long).to(self.device)
            with torch.no_grad():
                q_ = self.policy.q_target.forward(experience_batch.observation_next)
                max_actions = torch.argmax(self.policy.q_eval.forward(experience_batch.observation_next), dim=-1)
                target = experience_batch.reward + self._hp.discount_factor * q_[batch_idx, max_actions] * (
                        1 - experience_batch.done)
            q = self.policy.q_eval.forward(experience_batch.observation)[
                batch_idx, experience_batch.action_index.int()]

            loss = torch.nn.functional.smooth_l1_loss(q, target.detach())
            self._perform_update(loss, self.policy_opt, self.q_eval)

            # logging
            info = AttrDict(  # losses
                loss=loss,
                value=q.mean(),
                epsilon=self.policy.epsilon
            )

            info = map_dict(ten2ar, info)

        if self._update_steps % self._hp.target_update_interval == 0:
            self.policy.update_network_parameters()

        self._update_steps += 1
        self.policy.decrement_epsilon()
        return info

    def _act(self, obs, index=None, task=None):
        obs = map2torch(obs, self._hp.device)

        if len(obs.shape) == 1:  # we need batched inputs for policy
            policy_output = self._remove_batch(self.policy(obs[None]))
            return map2np(policy_output)
        return map2np(self.policy(obs))

    def _run_policy(self, obs):
        """Allows child classes to post-process policy outputs."""
        return self.policy(obs)

    def _act_rand(self, obs):
        policy_output = self.policy.sample_rand(map2torch(obs, self.policy.device))
        if hasattr(policy_output, 'dist'):
            del policy_output['dist']
        return map2np(policy_output)

    def _sample_experience(self):
        return self.replay_buffer.sample(n_samples=self._hp.batch_size)

    def _normalize_batch(self, experience_batch):
        """Optionally apply observation normalization."""
        return experience_batch

    def _split_image(self, obs):
        assert len(obs.shape) == 2 and obs.shape[1] == self._hp.policy_params.input_dim \
               + self._hp.policy_params.input_res ** 2 * 3
        return AttrDict(
            state=obs[:, :self._hp.policy_params.input_dim],
            image=obs[:, self._hp.policy_params.input_dim:].reshape(obs.shape[0], 3, self._hp.policy_params.input_res,
                                                                    self._hp.policy_params.input_res)
        )

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        d['policy_opt'] = self.policy_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.policy_opt.load_state_dict(state_dict.pop('policy_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def visualize(self, logger, rollout_storage, step):
        super().visualize(logger, rollout_storage, step)
        self.policy.visualize(logger, rollout_storage, step)

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer."""
        if not experience_batch:
            return  # pass if experience_batch is empty
        self.replay_buffer.append(experience_batch)

    def reset(self):
        self.policy.reset()

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        if len(experience_batch.observation_next.shape) == 2:
            experience_batch.observation_next = self._split_image(experience_batch.observation_next).image
            experience_batch.observation = self._split_image(experience_batch.observation).image
        return experience_batch
