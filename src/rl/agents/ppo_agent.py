import torch
from torch.nn import functional as F

from src.rl.components.agent import BaseAgent
from src.utils.general import AttrDict, map_dict
from src.utils.pytorch import map2np, map2torch, ten2ar, ar2ten
import numpy as np


class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self._hp = config
        self.policy = self._hp.policy(self._hp.policy_params)
        self.critic = self._hp.critic(self._hp.critic_params)

        # PPO 不用传统 replay buffer，用 rollout buffer
        self.rollout_buffer = self._hp.replay(self._hp.replay_params)
        self.policy_opt = self._get_optimizer(self._hp.optimizer, [self.policy, self.critic], self._hp.policy_lr)
        # self.critic_opt = self._get_optimizer(self._hp.optimizer, self.critic, self._hp.policy_lr)
        self.codebook = self._load_codebook()

        self.normalize_advantage = self._hp.normalize_advantage
        self.clip_epsilon = self._hp.clip_epsilon
        self.vf_coef = self._hp.vf_coef
        self.entropy_coef = self._hp.entropy_coef

    def update(self, experience_batch, info):

        self.add_experience(experience_batch)
        value = self.critic(self._split_image(map2torch(info.obs, self._hp.device).unsqueeze(0)).image)
        self.rollout_buffer.compute_returns_and_advantage(value, info.done)
        advantage_mean = self.rollout_buffer._rollout_buffer.advantage.mean()
        value_mean = self.rollout_buffer._rollout_buffer.value.mean()
        return_mean = self.rollout_buffer._rollout_buffer.return_.mean()
        log_prob_mean = self.rollout_buffer._rollout_buffer.log_prob.mean()

        for _ in range(self._hp.update_iterations):
            batch = self._sample_experience()
            batch = self._normalize_batch(batch)
            batch = map2torch(batch, self._hp.device)
            batch = self._preprocess_experience(batch)

            # get policy output
            policy_output = self.policy(batch.observation)
            # action = policy_output.action.cpu().numpy()
            log_prob = policy_output.log_prob
            entropy = policy_output.dist.entropy()
            value = self.critic(self._split_image(batch.observation).image)

            # compute ratio
            ratio = torch.exp(log_prob - batch.log_prob)

            # compute clipped surrogate loss
            advantage = batch.advantage
            if self.normalize_advantage and len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = F.mse_loss(value.squeeze(-1), batch.return_)

            entropy_loss = entropy.mean()

            # total loss
            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

            self._perform_update(loss, self.policy_opt, self)
            # self._perform_update(value_loss, self.critic_opt, self.critic)

            info = AttrDict(
                loss=loss,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy.mean(),
                log_prob=log_prob_mean,
                advantage_mean=advantage_mean,
                value_mean=value_mean,
                return_mean=return_mean,
            )
            info = map_dict(ten2ar, info)

        return info

    def _act(self, obs, index=None, task=None):
        obs = map2torch(obs, self._hp.device)
        policy_output = self.policy(obs)
        action = policy_output.action
        # value = self.critic(self._split_image(obs).image,
        #                     ar2ten(self.codebook[action, None], self._hp.device, torch.float)).q
        value = self.critic(self._split_image(obs).image)
        return map2np(
            AttrDict(action=self.codebook[action], action_index=action, log_prob=policy_output.log_prob, value=value))

    def _run_policy(self, obs):
        return self._act(obs)

    def _act_rand(self, obs):
        return self._act(obs)

    def _sample_experience(self):
        return self.rollout_buffer.sample(n_samples=self._hp.batch_size)

    def _normalize_batch(self, experience_batch):
        return experience_batch

    def _preprocess_experience(self, experience_batch):
        # if len(experience_batch.observation_next.shape) == 2:
        #     experience_batch.observation_next = self._split_image(experience_batch.observation_next).image
        #     experience_batch.observation = self._split_image(experience_batch.observation).image
        return experience_batch

    def _load_codebook(self):
        return np.eye(self._hp.policy_params.skill_dim)

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
        """Adds experience to rollout buffer."""
        if not experience_batch:
            return  # pass if experience_batch is empty
        self.rollout_buffer.append(experience_batch)
