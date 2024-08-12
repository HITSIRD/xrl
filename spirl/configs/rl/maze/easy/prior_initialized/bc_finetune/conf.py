from spirl.configs.rl.maze.easy.prior_initialized.base_conf import *
from spirl.rl.policies.prior_policies import PriorInitializedPolicy
from spirl.rl.agents.ac_agent import SACAgent

agent_config.policy = PriorInitializedPolicy
configuration.agent = SACAgent
