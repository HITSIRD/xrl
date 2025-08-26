import os
import copy

from src.models.bc import BCModel
from src.rl.agents.bc_agent import BCAgent
from src.rl.components.policy import Policy
from src.rl.components.buffer import UniformReplayBuffer
from src.utils.general import AttrDict
from src.rl.components.agent import FixedIntervalHierarchicalAgent
from src.rl.envs.kitchen import KitchenEnv
from src.rl.components.sampler import ACImageAugmentedHierarchicalSampler, HierarchicalSampler
from src.configs.default.fruits_snacks import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the kitchen env'

configuration = AttrDict(
    seed=42,
    agent=FixedIntervalHierarchicalAgent,
    environment=KitchenEnv,
    sampler=HierarchicalSampler,
    data_dir='.',
    num_epochs=10,
    max_rollout_len=9,
    n_steps_per_epoch=100000,
    n_warmup_steps=1000,
    n_steps_per_update=1,
    log_output_per_epoch=500,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=100000,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict()

sampler_config = AttrDict(
    n_frames=1,
)

base_agent_params = AttrDict(
    batch_size=128,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
    optimizer='adam',
    policy_lr=3e-4,
    adam_beta=0.9,
    gradient_clip=None,
    discount_factor=0.99,
)

###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    input_res=data_spec.res,
    action_dim=data_spec.n_actions,
    kl_div_weight=5e-4,
    prior_input_res=data_spec.res,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    nz_vae=10,
    n_input_frames=1,
    n_rollout_steps=10,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=BCModel,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill/real_kitchen/hierarchical"),
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,  # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    input_res=data_spec.res,
    max_action_range=2.,  # prior is Gaussian with unit variance
    unused_obs_size=60,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    input_res=hl_policy_params.input_res,
    output_dim=1,
    n_layers=2,  # number of policy network laye
    nz_mid=256,
    action_input=True,
    unused_obs_size=hl_policy_params.unused_obs_size,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=Policy,
    policy_params=hl_policy_params,
    critic_params=hl_critic_params,
))

##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=BCAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=BCAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=True,
    update_iterations=1,
    update_hl=True,
    update_ll=False,
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
