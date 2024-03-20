from spirl.configs.hrl.kitchen.spirl.conf import *
from spirl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.dqn_policy import DQNPolicy
from spirl.rl.agents.dqn_agent import DQNAgent

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/hierarchical_cl_vq/weights_K_32_logits"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = DQNPolicy

# update HL policy model params
hl_policy_params.update(AttrDict(
    policy=DQNPolicy,
    codebook_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                     "skill_prior_learning/kitchen/hierarchical_cl_vq/weights/weights_ep99.pth"),
    codebook_K=16,
    policy_lr=5e-5,
    target_update_interval=20,
    tau=0.1,
    epsilon=0.9,
    eps_end=0.01,
    eps_dec=1e-4,
    max_size=50000,
    batch_size=256,
    hidden_layers=[64, 64],
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    hl_agent=DQNAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))