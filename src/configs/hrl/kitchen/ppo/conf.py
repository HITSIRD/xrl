from d4rl.ope import normalize
from src.configs.hrl.kitchen.base_conf import *
from src.configs.hrl.kitchen.base_conf import hl_critic_params
from src.configs.skill.kitchen.prior_bc.conf import model_config
from src.models.bc import OneHotImagePriorBCModel
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.components.buffer import RolloutBuffer
from src.rl.components.critic import CNNCritic
from src.rl.policies.ac_policy import ACPriorInitializedPolicy
from src.rl.policies.cl_model_policy import ACClModelPolicy
from stable_baselines3.ppo import PPO

epoch = 9

configuration.update(AttrDict(
    sampler=ACImageAugmentedHierarchicalSampler,
    n_steps_per_update=1000,
    n_val_sample=10,
    n_warmup_steps=0,
))

replay_params.update(AttrDict(
    capacity=configuration.n_steps_per_update,
    gae_lambda=0.95,
    discount_factor=base_agent_params.discount_factor,
))

ll_model_params.update(model_config)
ll_model_params.update(AttrDict(
    cond_decode=True,
    update_encoder=False,
    # if_freeze=False,
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    load_weights=True,
    policy_model_epoch=epoch,
    initial_log_sigma=-50,
    policy_model=OneHotImagePriorBCModel,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill/kitchen/prior_bc/resnet-mix"),
)

ll_policy_params.update(ll_model_params)

ll_agent_config.update(AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
))

# update HL policy model params
hl_policy_params.update(AttrDict(
    load_weights=True,
    policy_model_epoch=0,
    policy_model_params=ll_policy_params.policy_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "hrl/kitchen/ppo/mix-kbts_s0/weights"),

    prior_model_epoch=epoch,
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,

    img_enc_dim=ll_policy_params.img_enc_dim,
    skill_dim=ll_policy_params.skill_dim,
))

hl_critic_params.update(AttrDict(
    output_dim=1,
    action_dim=hl_policy_params.skill_dim,
    input_res=data_spec.res,
    img_enc_dim=128,
    nz_mid=256,
    unused_obs_size=60,
    use_resnet=ll_model_params.use_resnet,
))

hl_agent_config.update(AttrDict(
    policy=ACPriorInitializedPolicy,
    critic=CNNCritic,
    policy_params=hl_policy_params,
    critic_params=hl_critic_params,
    replay=RolloutBuffer,
    replay_params=replay_params,
    update_iterations=10,
    policy_lr=2e-4,

    gae_lambda=replay_params.gae_lambda,
    batchsize=128,
    clip_epsilon=0.2,
    vf_coef=0.5,
    entropy_coef=0,
    normalize_advantage=True,
))

agent_config.update(AttrDict(
    hl_agent=PPOAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=BCAgent,
    ll_agent_params=ll_agent_config,
    update_iterations=hl_agent_config.update_iterations,
    hl_interval=10,
))
