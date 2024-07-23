from spirl.configs.hrl.calvin.spirl.conf import *
from spirl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.deterministic_policies import DeterministicPolicy
from spirl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy
from spirl.rl.agents.prior_discrete_sac_agent import ActionPriorDiscreteSACAgent
from spirl.rl.agents.discrete_ac_agent import DiscreteSACAgent

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

ll_model_params.update(AttrDict(
    codebook_K=8,
    # fixed_codebook=False,
))

# Q(s) instead of Q(s, a)
# hl_critic_params.update(AttrDict(
#     input_dim=hl_policy_params.input_dim,
#     output_dim=ll_model_params.codebook_K,
#     action_input=False,
# ))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/calvin/hierarchical_cl_vq/K_8"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,  # ClModelPolicy
    policy_params=ll_policy_params,
    critic=MLPCritic,                   # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = DeterministicPolicy

# update HL policy model params
hl_policy_params.update(AttrDict(
    policy=DeterministicPolicy,
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    # policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                      "hrl/calvin/spirl_cl_vq/k8_s6/weights"),
    # squash_output_dist=False,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))

