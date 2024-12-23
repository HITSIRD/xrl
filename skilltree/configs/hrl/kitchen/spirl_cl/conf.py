from skilltree.configs.hrl.kitchen.spirl.conf import *
from skilltree.models.closed_loop_spirl_mdl import ClSPiRLMdl, ImageClSPiRLMdl
from skilltree.rl.policies.cl_model_policies import ClModelPolicy, ACClModelPolicy
from skilltree.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ImageClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/hierarchical_cl"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,                   # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = ACLearnedPriorAugmentedPIPolicy

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    # policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                      "hrl/kitchen/spirl_cl/mlsh_s3"),
    # policy_model_epoch=24,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))
