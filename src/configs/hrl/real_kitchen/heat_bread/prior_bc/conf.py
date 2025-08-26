from src.configs.hrl.real_kitchen.heat_bread.base_conf import *
from src.configs.skill.real_kitchen.heat_bread.prior_bc.conf import model_config
from src.models.bc import OneHotImagePriorBCModel
from src.rl.components.sampler import ImageAugmentedSampler
from src.rl.policies.deterministic_policy import DeterministicPolicy, PriorDeterministicPolicy
from src.rl.policies.script_policy import ScriptPolicy

epoch = 9

configuration.update(AttrDict(
    sampler=ImageAugmentedSampler,
    n_val_sample=20,
))

ll_model_params.update(model_config)
ll_model_params.update(AttrDict(
    cond_decode=True,
    update_encoder=False,
    # if_freeze=False,
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    # policy_model=ImageClVQCDTMdl,
    load_weights=True,
    policy_model_epoch=epoch,
    initial_log_sigma=-50,
    policy_model=OneHotImagePriorBCModel,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill/real_kitchen/prior_bc/heat_bread/top50"),
)

ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config.update(AttrDict(
    policy=ScriptPolicy,
    policy_params=ll_policy_params,
))

# update HL policy model params
hl_policy_params.update(AttrDict(
    load_weights=True,
    prior_model_epoch=epoch,
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

hl_agent_config.update(AttrDict(
    policy=PriorDeterministicPolicy,
    policy_params=hl_policy_params,
    # critic=SplitObsMLPCritic,
    # critic_params=hl_critic_params,
))

agent_config.update(AttrDict(
    hl_agent=BCAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=BCAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=1,
))
