from spirl.configs.hrl.kitchen.spirl.conf import *
from spirl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl, ClVQSPiRLMdlExtension
from spirl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl, ClVQCDTMdlExtension
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy, LearnedVQPriorAugmentedPolicyCDT

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

prior_model_name = "spirl_k16_s2_1"

# CDT config
ll_model_params.update(AttrDict(
    codebook_K = 16,
    feature_learning_depth = -1,
    num_intermediate_variables = 20,
    decision_depth = 4,
    greatest_path_probability = 0,
    beta_fl = 0,
    beta_dc = 0,
    if_smooth = False,
    if_save = False,
    tree_name = "mlsh",
    if_freeze=False,
    cdt_embedding_checkpoint=os.path.join(os.environ["EXP_DIR"], 
                                          f"skill_prior_learning/kitchen/hierarchical_cl_vq_extension/{prior_model_name}/weights"),
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQSPiRLMdlExtension,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         f"skill_prior_learning/kitchen/hierarchical_cl_vq_extension/{prior_model_name}"),
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
    policy=LearnedVQPriorAugmentedPolicyCDT,
    policy_model=ClVQCDTMdlExtension,
    policy_model_params=ll_policy_params.policy_model_params,
    load_weights=False, # 不使用先验初始化
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    squash_output_dist=False,   # TODO fa7475f：保持对数概率的原始值？
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))

agent_config.hl_agent_params.update(AttrDict(   # TODO fa7475f：某个参数？
    td_schedule_params=AttrDict(p=1.5),
))