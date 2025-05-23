import torch

from src.rl.components.agent import BaseAgent
from src.rl.components.policy import Policy
from src.utils.general import AttrDict
from src.utils.pytorch import no_batchnorm_update


class ACPriorInitializedPolicy(Policy):
    """Initializes policy network with learned prior net."""

    def __init__(self, config):
        self._hp = config
        self.update_model_params(self._hp.prior_model_params)
        super().__init__()

        if hasattr(self._hp, 'policy_model_checkpoint') and self._hp.policy_model_checkpoint is not None:
            print('load high level policy from {}'.format(self._hp.policy_model_checkpoint))
            BaseAgent.load_model_weights(self.net, self._hp.policy_model_checkpoint, self._hp.policy_model_epoch,
                                     opt='hl_agent')

    def forward(self, obs):
        with no_batchnorm_update(self):  # BN updates harm the initialized policy
            return super().forward(obs)

    def _build_network(self):
        net = self._hp.prior_model(self._hp.prior_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.prior_model_epoch)
        return net

    def _compute_action_dist(self, obs):
        return self.net.compute_learned_prior(self._split_obs(obs).prior_obs)

    def _split_obs(self, obs):
        if isinstance(obs, AttrDict):
            return AttrDict(
                cond_input=self.net.enc_obs(obs.obs),
                z=obs.hl_action,
            )
        else:
            unflattened_obs = self.net.unflatten_obs(obs)
            return unflattened_obs

    def sample_rand(self, obs):
        if isinstance(obs, AttrDict):
            return self.forward(obs)
        if len(obs.shape) == 1:
            output_dict = self.forward(obs[None])
            output_dict.action = output_dict.action[0]
            return output_dict
        return self.forward(obs)  # for prior-initialized policy we run policy directly for rand sampling from prior

    @staticmethod
    def update_model_params(params):
        params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # params.batch_size = 1  # run only single-element batches for forward pass
