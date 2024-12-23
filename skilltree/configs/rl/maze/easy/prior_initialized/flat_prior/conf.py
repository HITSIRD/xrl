from strl.configs.rl.maze.prior_initialized.base_conf import *
from strl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from strl.data.maze.src.maze_agents import MazeActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = ACLearnedPriorAugmentedPIPolicy
configuration.agent = MazeActionPriorSACAgent
