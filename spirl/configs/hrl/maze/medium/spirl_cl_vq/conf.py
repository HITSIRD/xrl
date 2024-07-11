from spirl.configs.hrl.maze.medium.spirl.conf import *
from spirl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from spirl.rl.components.critic import MLPCritic
from spirl.rl.policies.cl_model_policies import ClModelPolicy
from spirl.rl.policies.prior_policies import LearnedVQPriorAugmentedPolicy

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': ACRandMaze0S30Env,
    # 'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 25,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 1000,
}
configuration = AttrDict(configuration)

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

ll_model_params.update(AttrDict(
    codebook_K=8,
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=ClVQSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/maze/medium/hierarchical_cl_vq/K_8"),
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,  # ClModelPolicy
    policy_params=ll_policy_params,
    critic=MLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

hl_agent_config.policy = LearnedVQPriorAugmentedPolicy

# update HL policy model params
hl_policy_params.update(AttrDict(
    policy=LearnedVQPriorAugmentedPolicy,  # PriorInitializedPolicy PriorAugmentedPolicy
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    squash_output_dist=False,
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

agent_config.hl_agent_params.update(AttrDict(
    td_schedule_params=AttrDict(p=1.5),
))
