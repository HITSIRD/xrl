import copy

from strl.configs.rl.maze.easy.base_conf import *
from strl.rl.components.sampler import ACMultiImageAugmentedSampler
from strl.rl.policies.mlp_policies import ConvPolicy
from strl.rl.components.critic import SplitObsMLPCritic
from strl.models.bc_mdl import BCMdl

# update sampler
# configuration['sampler'] = ACMultiImageAugmentedSampler
# sampler_config = AttrDict(
#     n_frames=2,
# )
env_config.screen_width = data_spec.res
env_config.screen_height = data_spec.res

prior_model_params=AttrDict(state_dim=data_spec.state_dim,
                                action_dim=data_spec.n_actions,
                                input_res=data_spec.res,
                                )

# update policy to conv policy
agent_config.policy = ConvPolicy
policy_params.update(AttrDict(
    prior_model=BCMdl,
    prior_model_params=copy.deepcopy(prior_model_params),
    prior_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                        "skill_prior_learning/maze/easy/flat"),
    # policy_model=BCMdl,
    # policy_model_params=copy.deepcopy(prior_model_params),
    # policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                      "rl/maze/easy/prior_initialized/bc_finetune/s4"),
    # policy_model_epoch=14,
))

# update critic+policy to handle multi-frame combined observation
# agent_config.critic = SplitObsMLPCritic
# agent_config.critic_params.unused_obs_size = 32 ** 2 * 3 * sampler_config.n_frames
