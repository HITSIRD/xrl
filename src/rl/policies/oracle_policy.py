import torch
import numpy as np

from src.utils.general import AttrDict, ParamDict
from src.rl.components.agent import BaseAgent
from src.rl.components.policy import Policy
from src.utils.pytorch import no_batchnorm_update


class OraclePolicy(Policy):
    def __init__(self, config):
        self._hp = config
        # self.update_model_params(self._hp.prior_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None
        self.skill_enc = np.eye(7)
        # self.goal = [5, 6, 0, 2]
        # self.goal = [5, 2, 3, 4]
        self.goal = [6, 0, 1, 3]
        self.current_goal_idx = 0
        # self.codebook = self._load_codebook()

    def forward(self, obs, index=None):
        obs = self._split_obs(obs)
        skill = (torch.from_numpy(self.skill_enc[self.goal[self.current_goal_idx]])
                 .float().unsqueeze(0).to(self.device))
        img_embd = self.net.complete_encoder(obs.images)
        logits = self.net.classifier(torch.cat([img_embd, skill], -1))
        self.current_goal_idx = min(self.current_goal_idx + (torch.sigmoid(logits) > 0.1), len(self.goal) - 1)
        # print(torch.sigmoid(logits))
        index = self.goal[self.current_goal_idx]
        return AttrDict(action=self.skill_enc[index], action_index=index)

    def _build_network(self):
        self._hp.prior_model_params.device = self._hp.device
        net = self._hp.prior_model(self._hp.prior_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.prior_model_epoch)
        return net

    def _compute_action_dist(self, obs):
        return self.net.compute_learned_prior(obs, first_only=True)

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None
        self.current_goal_idx = 0

    def _split_obs(self, obs):
        if isinstance(obs, AttrDict):
            return AttrDict(
                cond_input=self.net.enc_obs(obs.obs),
                z=obs.hl_action,
            )
        else:
            unflattened_obs = self.net.unflatten_obs(obs)
            return AttrDict(
                images=unflattened_obs.prior_obs,
                skills=obs[:, -self.net.latent_dim:],
            )

    @staticmethod
    def update_model_params(params):
        params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        params.batch_size = 1  # run only single-element batches for forward pass

    @property
    def horizon(self):
        return self._hp.policy_model_params.n_rollout_steps

    @property
    def has_trainable_params(self):
        """Indicates whether policy has trainable params."""
        return False


class PriorOraclePolicy(OraclePolicy):
    def _build_network(self):
        self._hp.prior_model_params.device = self._hp.device
        net = self._hp.prior_model(self._hp.prior_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.prior_model_epoch)
        return net

    def forward(self, obs, index=None):
        obs = self._split_obs(obs)
        index = torch.argmax(self.net.prior_head(self.net.prior_encoder(obs.images))).detach().cpu().item()
        return AttrDict(action=self.skill_enc[index], action_index=index)

    def _split_obs(self, obs):
        if isinstance(obs, AttrDict):
            return AttrDict(
                cond_input=self.net.enc_obs(obs.obs),
                z=obs.hl_action,
            )
        else:
            unflattened_obs = self.net.unflatten_obs(obs)
            return AttrDict(
                images=unflattened_obs.prior_obs,
                skills=obs[:, -self.net.latent_dim:],
            )
