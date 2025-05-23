import torch
import torch.nn as nn
import copy

from src.utils.general import ParamDict, AttrDict
from src.modules.networks import MLPEncoder, CNNEncoder


class Critic(nn.Module):
    """Base critic class."""

    def __init__(self):
        super().__init__()

    def forward(self, obs, actions=None):
        raise NotImplementedError("Needs to be implemented by child class.")

    @staticmethod
    def dummy_output():
        return AttrDict(q=None)


class MLPCritic(Critic):
    """MLP-based critic."""

    def __init__(self, config):
        self._hp = config
        super().__init__()
        self._net = self._build_network()

    def forward(self, obs, actions=None):
        input = torch.cat((obs, actions), dim=-1)
        return AttrDict(q=self._net(input))

    def _build_network(self):
        input_size = self._hp.input_dim + self._hp.action_dim
        return torch.nn.Sequential(
            MLPEncoder(input_size, self._hp.hidden_dims, self._hp.hidden_dims),
            nn.Linear(self._hp.hidden_dims[-1], 1)
        )


class CNNCritic(Critic):
    """Critic that can incorporate image and action inputs by fusing conv and MLP encoder."""

    def __init__(self, config):
        self._hp = config
        super().__init__()

        self.head = nn.Sequential(
            # nn.Linear(self._hp.img_enc_dim + self._hp.action_dim, self._hp.nz_mid),
            nn.Linear(self._hp.img_enc_dim, self._hp.nz_mid),
            nn.ReLU(),
            nn.Linear(self._hp.nz_mid, self._hp.output_dim),
        )

        self.encoder = CNNEncoder(3, self._hp.input_res, self._hp.img_enc_dim)

    def forward(self, obs, actions=None):
        image = obs.reshape(-1, 3, self._hp.input_res, self._hp.input_res)
        img_enc = self.encoder(image)
        # q = self.head(torch.cat((img_enc, actions), dim=-1))
        v = self.head(img_enc)
        return v

# class HybridCritic(ConvCritic):
#     def _build_network(self):
#         return HybridCriticEncoder(copy.deepcopy(self._hp))
#
#     def forward(self, obs, actions):
#         input = AttrDict(
#             states=obs.states,
#             actions=actions,
#             images=obs.images
#         )
#         return AttrDict(q=self._net(input))
