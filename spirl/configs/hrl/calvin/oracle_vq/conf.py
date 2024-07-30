from spirl.configs.hrl.calvin.spirl.conf import *
from spirl.rl.components.sampler import Sampler
from spirl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.oracle_vq_policies import OracleVQPolicy

configuration.update(AttrDict(sampler=Sampler))

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

hl_agent_config.policy = OracleVQPolicy

ll_model_params.update(AttrDict(
    codebook_K=16,
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/calvin/hierarchical_cl_vq/K_16"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,

    codebook_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                     "skill_prior_learning/calvin/hierarchical_cl_vq/K_16/weights/weights_ep99.pth"),
    skill_evaluation=os.path.join(os.environ["EXP_DIR"], "hrl/calvin/oracle_vq/K_16/skill_evaluate_prior_")
))

agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))
