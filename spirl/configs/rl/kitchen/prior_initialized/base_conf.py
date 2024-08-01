from spirl.configs.rl.kitchen.base_conf import *
from spirl.models.bc_mdl import BCMdl

policy_params.update(AttrDict(
    prior_model=BCMdl,
    prior_model_params=AttrDict(state_dim=data_spec.state_dim,
                                action_dim=data_spec.n_actions,
                        ),
    prior_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/kitchen/flat"),
    squash_output_dist=False,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "rl/kitchen/prior_initialized/bc_finetune/mkbl_s0"),
    prior_model_epoch=19,
))