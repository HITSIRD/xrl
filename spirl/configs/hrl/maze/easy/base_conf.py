import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.policies.mlp_policies import SplitObsMLPPolicy
from spirl.rl.components.critic import SplitObsMLPCritic, MLPCritic
from spirl.rl.envs.maze import ACRandMaze0S20Env
from spirl.rl.components.sampler import ACMultiImageAugmentedHierarchicalSampler, HierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from spirl.configs.default_data_configs.maze import data_spec
from spirl.data.maze.src.maze_agents import MazeACSkillSpaceAgent

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the maze env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': ACRandMaze0S20Env,
    # 'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 25,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 1000,
}
configuration = AttrDict(configuration)

# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
)

###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-3,
    nz_env=128,
    nz_mid=128,
    n_processing_layers=5,
    nz_vae=10,
    prior_input_res=data_spec.res,
    # n_input_frames=2,
    cond_decode=True,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=ImageSkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/maze/hierarchical"),
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,  # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,  # prior is Gaussian with unit variance
    # unused_obs_size=ll_model_params.prior_input_res ** 2 * 3 * ll_model_params.n_input_frames,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    # unused_obs_size=hl_policy_params.unused_obs_size,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=SplitObsMLPPolicy,
    policy_params=hl_policy_params,
    # critic=SplitObsMLPCritic,
    critic=MLPCritic,
    critic_params=hl_critic_params,
))

##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=MazeACSkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=False,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)

# reduce replay capacity because we are training image-based, do not dump (too large)
from spirl.rl.components.replay_buffer import SplitObsUniformReplayBuffer

agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = ll_model_params.prior_input_res ** 2 * 3 * 2 + \
                                                             hl_agent_config.policy_params.action_dim  # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False
