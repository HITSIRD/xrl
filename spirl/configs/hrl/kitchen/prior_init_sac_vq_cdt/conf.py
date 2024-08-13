from spirl.configs.hrl.kitchen.spirl.conf import *
from spirl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from spirl.models.closed_loop_vq_cdt_mdl import ClVQCDTMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy, PriorInitializedPolicy, LearnedVQPriorAugmentedPolicyCDT
from spirl.rl.agents.discrete_ac_agent import DiscreteSACAgent

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

prior_model_name = "cdt_k16_s1_-1+60+6+0_1"

# CDT config
ll_model_params.update(AttrDict(
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
    # if_freeze=False,
    # cdt_embedding_checkpoint=os.path.join(os.environ["EXP_DIR"], 
    #   f"skill_prior_learning/kitchen/hierarchical_cl_vq_cdt/{prior_model_name}/weights"), // 其它组件的位置
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQCDTMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         f"skill_prior_learning/kitchen/hierarchical_cl_vq_cdt/{prior_model_name}"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = PriorInitializedPolicy


# update HL policy model params 
hl_policy_params.update(AttrDict(
    policy=PriorInitializedPolicy
,  # PriorInitializedPolicy PriorAugmentedPolicy
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    squash_output_dist=False,  # TODO fa7475f：保持对数概率的原始值？
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))
