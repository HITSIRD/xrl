from spirl.configs.rl.calvin.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import PriorInitializedPolicy

agent_config.policy = PriorInitializedPolicy
configuration.agent = SACAgent