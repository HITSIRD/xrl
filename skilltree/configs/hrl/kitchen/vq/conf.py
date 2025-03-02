from skilltree.configs.hrl.kitchen.spirl.conf import *
from skilltree.models.closed_loop_vq_cdt_mdl import ImageClVQCDTMdl
from skilltree.rl.components.sampler import Sampler, ImageAugmentedSampler
from skilltree.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl, ImageClVQSPiRLMdl
from skilltree.rl.policies.cl_model_policies import ClModelPolicy, ACClModelPolicy
from skilltree.rl.policies.deterministic_policies import DeterministicPolicy

configuration.update(AttrDict(sampler=ImageAugmentedSampler))

# update model params to conditioned decoder on state
ll_model_params.cond_decode = True

hl_agent_config.policy = DeterministicPolicy

# CDT config
ll_model_params.update(AttrDict(
    codebook_K = 16,
    fixed_codebook=False,
    feature_learning_depth = -1,
    num_intermediate_variables = 20,
    decision_depth = 6,
    greatest_path_probability = 1,
    beta_fl = 0,
    beta_dc = 0,
    if_smooth = False,
    if_save = False,
    tree_name = "",
    update_encoder=False,
    # if_freeze=False,
    # cdt_embedding_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                        #   f"skill_prior_learning/kitchen/hierarchical_cl_vq_cdt/{prior_model_name}/weights"), // 其它组件的位置
))

# create LL closed-loop policy
ll_policy_params = AttrDict(
    # policy_model=ImageClVQCDTMdl,
    policy_model=ImageClVQSPiRLMdl,
    policy_model_params=ll_model_params,
    # policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                      "skill_prior_learning/kitchen/hierarchical_cl_vq"),
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/hierarchical_cl_vq"),
)

ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,  # LL critic is not used since we are not finetuning LL
    critic_params=hl_critic_params
)

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
    codebook_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                     "hrl/kitchen/cdt_cl_vq_prior_cdt/mkbl_d6_s1_avgprob/weights/weights_ep9.pth"),
    # codebook_checkpoint=os.path.join(os.environ["EXP_DIR"],
    #                                  "skill_prior_learning/kitchen/hierarchical_cl_vq/weights/weights_ep99.pth"),
))

agent_config.update(AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))
