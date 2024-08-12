from spirl.configs.hrl.maze.easy.spirl.conf import *
from spirl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl
from spirl.rl.components.critic import MLPCritic
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy, LearnedVQPriorAugmentedPolicyCDT

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': ACRandMaze0S20Env,
    # 'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 21,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 1000,
}
configuration = AttrDict(configuration)

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

prior_model_name = "easy_cdt_k16_s1_-1+20+5+0_1"
prior_model_epoch = 74

# CDT config
ll_model_params.update(AttrDict(
    codebook_K = 16,
    fixed_codebook = False,
    feature_learning_depth = -1,
    num_intermediate_variables = 20,
    decision_depth = 5,
    greatest_path_probability = 1,
    beta_fl = 0,
    beta_dc = 0,
    if_smooth = False,
    if_save = False,
    tree_name = "mlsh_cdtk162_-1+60+5_n+b_s9_1"
    # if_freeze=False,
    # cdt_embedding_checkpoint=os.path.join(os.environ["EXP_DIR"], 
                                        #   f"skill_prior_learning/kitchen/hierarchical_cl_vq_cdt/{prior_model_name}/weights"), // 其它组件的位置
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQCDTMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         f"skill_prior_learning/maze/easy/hierarchical_cl_vq_cdt/{prior_model_name}"),
    policy_model_epoch=prior_model_epoch,
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                   # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = LearnedVQPriorAugmentedPolicyCDT

# update HL policy model params 
hl_policy_params.update(AttrDict(
    policy=LearnedVQPriorAugmentedPolicy, # PriorInitializedPolicy PriorAugmentedPolicy 
    prior_model=ll_policy_params.policy_model, 
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,    
    prior_model_epoch=prior_model_epoch,
    squash_output_dist=False,   # TODO fa7475f：保持对数概率的原始值？
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    log_videos=False,
    update_hl=True,
    update_ll=False,
))

agent_config.hl_agent_params.update(AttrDict(   # TODO fa7475f：某个参数？
    td_schedule_params=AttrDict(p=5.0),
))