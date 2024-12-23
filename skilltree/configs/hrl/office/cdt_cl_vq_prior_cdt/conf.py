import os
import copy

from strl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from strl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl
from strl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from strl.utils.general_utils import AttrDict
from strl.rl.components.agent import FixedIntervalHierarchicalAgent
from strl.rl.components.critic import MLPCritic
from strl.rl.envs.office import OfficeEnv
from strl.rl.components.sampler import HierarchicalSampler
from strl.rl.components.replay_buffer import UniformReplayBuffer
from strl.rl.agents.ac_agent import SACAgent
from strl.rl.policies.cl_model_policies import ClModelPolicy
from strl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy, LearnedVQPriorAugmentedPolicyCDT
from strl.configs.default_data_configs.office import data_spec

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the office env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': OfficeEnv,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 16,
    'max_rollout_len': 350,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 2e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

base_agent_params = AttrDict(
    batch_size=10, #256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
prior_model_name = "cdt_k16_s1_-1+6+0_1"

# LL Policy Params
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    nz_vae=10,
    nz_mid=128,
    n_processing_layers=5,
    kl_div_weight=5e-4,
    cond_decode=True,
    codebook_K=16,

    fixed_codebook=False,
    feature_learning_depth=-1,
    num_intermediate_variables=20,
    decision_depth=6,
    greatest_path_probability=1,
    beta_fl=0,
    beta_dc=0,
    if_smooth=False,
    if_save=False,
    tree_name="mlsh_cdtk162_-1+60+6_n+b_s9_1"
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ClVQCDTMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         f"skill_prior_learning/office/hierarchical_cl_vq_cdt/{prior_model_name}"),
    initial_log_sigma=-50,
    # policy_model_epoch=99,
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    output_dim=1,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                           # LL critic is not used since we are not finetuning LL
    critic_params=ll_critic_params,
))

###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    policy=LearnedVQPriorAugmentedPolicy,
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    # prior_model_epoch=ll_policy_params.policy_model_epoch,
    squash_output_dist=False,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,
    nz_mid=256,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=LearnedVQPriorAugmentedPolicyCDT,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
    td_schedule_params=AttrDict(p=1.),
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_video_caption=False,
    update_hl=True,
    update_ll=False,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    name='Widow250OfficeFixed-v0',
)

