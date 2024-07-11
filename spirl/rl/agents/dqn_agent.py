import torch
import numpy as np

from spirl.rl.components.agent import BaseAgent
from spirl.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
from spirl.utils.general_utils import ParamDict, map_dict, AttrDict


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = self._default_hparams().overwrite(config)
        self.policy = self._hp.policy(self._hp.policy_params)

        # build replay buffer
        self.replay_buffer = self._hp.replay(self._hp.replay_params)
        self.policy_opt = self._get_optimizer(self._hp.optimizer, self.policy.q_eval, self._hp.policy_lr)
        self._update_steps = 0

        if not self._hp.update_codebook:
            self.policy.codebook.requires_grad = False

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy_lr': 5e-5,  # learning rate for policy update
            'update_codebook': False,
            'target_update_interval': 100,
        })
        return super()._default_hparams().overwrite(default_dict)

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
                # done_index = torch.where(experience_batch.done > 0.5)
                # if not done_index:
                #     q_[done_index] = 0.0
                target = experience_batch.reward + self._hp.discount_factor * q_[batch_idx, max_actions] * (
                            1 - experience_batch.done)
            q = self.policy.q_eval.forward(experience_batch.observation)[
                batch_idx, experience_batch.action_index.long()]

            mse_loss = torch.nn.MSELoss()
            loss = torch.nn.functional.smooth_l1_loss(q, target.detach())
            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()

            # logging
            info = AttrDict(  # losses
                loss=loss,
                value=q.mean(),
                epsilon=self.policy.epsilon
            )
            # if self._update_steps % 100 == 0:
            #     info.update(AttrDict(       # gradient norms
            #         policy_grad_norm=avg_grad_norm(self.policy),
            #         critic_1_grad_norm=avg_grad_norm(self.critics[0]),
            #         critic_2_grad_norm=avg_grad_norm(self.critics[1]),
            #     ))

            info = map_dict(ten2ar, info)

        if self._update_steps % self._hp.target_update_interval == 0:
            self.policy.update_network_parameters()
            # print(f'steps: {self._update_steps}')
            # print(f'loss: {loss}')
            # print(f'q: {q.mean()}')

        self._update_steps += 1
        self.policy.decrement_epsilon()
        return info

    def _act(self, obs, index=None, task=None):
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)

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
        experience_batch.observation = self._obs_normalizer(experience_batch.observation)
        experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
        return experience_batch

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
        self._obs_normalizer.update(experience_batch.observation)

    def reset(self):
        self.policy.reset()

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        return experience_batch
