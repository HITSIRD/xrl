import stable_baselines3.dqn

from strl.rl.components.agent import BaseAgent
from strl.rl.policies.dqn_policies import DQNPolicy
from strl.rl.policies.prior_policies import PriorInitializedPolicy
from strl.utils.pytorch_utils import no_batchnorm_update
stable_baselines3


class PriorWarmupDQNPolicy(DQNPolicy):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        DQNPolicy.__init__(self, config)

    def _build_network(self):
        net = self._hp.prior_model(self._hp.prior_model_params, None)
        # BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.prior_model_epoch)
        BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, 99)
        return net

    def sample_rand(self, obs):
        if len(obs.shape) == 1:
            output_dict = self.forward(obs[None])
            output_dict.action = output_dict.action[0]
            return output_dict
        with no_batchnorm_update(self):
            return super().sample_rand(obs, prior=True)

    def _compute_action_dist(self, obs):
        return self.net.compute_learned_prior(obs, first_only=True)
