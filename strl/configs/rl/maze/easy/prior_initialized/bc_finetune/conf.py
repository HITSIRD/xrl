from strl.configs.rl.maze.easy.prior_initialized.base_conf import *
from strl.rl.policies.prior_policies import PriorInitializedPolicy
from strl.rl.agents.ac_agent import SACAgent

agent_config.policy = PriorInitializedPolicy
configuration.agent = SACAgent
