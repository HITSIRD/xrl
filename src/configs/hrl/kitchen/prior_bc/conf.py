from src.configs.hrl.kitchen.base_conf import *
from src.configs.skill.kitchen.prior_bc.conf import model_config
from src.models.bc import OneHotImagePriorBCModel
from src.rl.components.sampler import Sampler, ImageAugmentedSampler
from src.rl.policies.cl_model_policy import ACClModelPolicy
from src.rl.policies.deterministic_policy import DeterministicPolicy, PriorDeterministicPolicy

epoch=0

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
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill/kitchen/prior_bc"),
)

ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config.update(AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
    # critic=SplitObsMLPCritic,  # LL critic is not used since we are not finetuning LL
    # critic_params=hl_critic_params
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
))
