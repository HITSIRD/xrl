from src.configs.hrl.kitchen.base_conf import *
from src.configs.skill.kitchen.prior_bc.conf import model_config
from src.models.bc import OneHotImagePriorBCModel
from src.rl.agents.dqn_agent import DQNAgent
from src.rl.policies.cl_model_policy import ACClModelPolicy
from src.rl.policies.dqn_policy import DQNPolicy

epoch = 0

configuration.update(AttrDict(
    sampler=ACImageAugmentedHierarchicalSampler,
    n_val_sample=10,
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
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill/kitchen/prior_bc"),
)

ll_policy_params.update(ll_model_params)

ll_agent_config.update(AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
))

# update HL policy model params
hl_policy_params.update(AttrDict(
    load_weights=True,
    prior_model_epoch=epoch,
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    img_enc_dim=ll_policy_params.img_enc_dim,
    skill_dim=ll_policy_params.skill_dim,
    input_res=data_spec.res,
    epsilon=1.0,
    tau=1.0,
    eps_decay=0.999,
    eps_min=0.005,
))

hl_agent_config.update(AttrDict(
    policy=DQNPolicy,
    policy_params=hl_policy_params,
    update_iterations=1,
    target_update_interval=500,
    target_network_update_factor=5e-3,
))

agent_config.update(AttrDict(
    hl_agent=DQNAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=BCAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=10,
))
