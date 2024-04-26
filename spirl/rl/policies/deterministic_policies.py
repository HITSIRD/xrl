import torch
import numpy as np

from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.rl.components.agent import BaseAgent
from spirl.rl.components.policy import Policy


class DeterministicPolicy(Policy):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.update_model_params(self._hp.prior_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy_model': None,  # policy model class
            'policy_model_params': None,  # parameters for the policy model
            'policy_model_checkpoint': None,  # checkpoint path of the policy model
            'policy_model_epoch': 'latest',  # epoch that checkpoint should be loaded for (defaults to latest)
            'load_weights': True,  # optionally allows to *not* load the weights (ie train from scratch)
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs, index):
        index = np.random.randint(16)
        return AttrDict(action=self.net[index].detach().cpu().numpy(), action_index=index)

    def _build_network(self):
        # net = self._hp.prior_model(self._hp.prior_model_params, None)
        weight = torch.load(self._hp.codebook_checkpoint)
        # return weight['state_dict']['hl_agent']['policy.prior_net.codebook.embedding.weight']
        # return weight['state_dict']['hl_agent']['policy.net.codebook.embedding.weight']
        return weight['state_dict']['codebook.embedding.weight']

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

    def _split_obs(self, obs):
        assert obs.shape[1] == self.net.state_dim + self.net.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self.net.latent_dim],  # condition decoding on state
            z=obs[:, -self.net.latent_dim:],
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
