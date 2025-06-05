import torch
import numpy as np

from src.utils.general import AttrDict, ParamDict
from src.rl.components.agent import BaseAgent
from src.rl.components.policy import Policy


class DeterministicPolicy(Policy):
    def __init__(self, config):
        self._hp = config
        # self.update_model_params(self._hp.prior_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None
        self.skill_enc = np.eye(7)
        # self.codebook = self._load_codebook()

    def forward(self, obs, index=None):
        assert index is not None
        return AttrDict(action=self.skill_enc[index], action_index=index)

    def _build_network(self):
        return None

    def _load_codebook(self):
        weight = torch.load(self._hp.codebook_checkpoint)
        print('loading codebook from {}'.format(self._hp.codebook_checkpoint))

        # return weight['state_dict']['hl_agent']['policy.prior_net.codebook.embedding.weight'].cpu().numpy()
        # return weight['state_dict']['hl_agent']['policy.net.codebook.embedding.weight'].cpu().numpy()
        return weight['state_dict']['codebook.embedding.weight'].cpu().numpy()

    def _compute_action_dist(self, obs):
        return self.net.compute_learned_prior(obs, first_only=True)

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

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


class PriorDeterministicPolicy(DeterministicPolicy):
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
