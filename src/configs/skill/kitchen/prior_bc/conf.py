from src.configs.default.kitchen.conf import data_spec
from src.configs.skill.kitchen.bc.conf import *
from src.models.bc import OneHotImagePriorBC
from src.utils.general import AttrDict

configuration.update(AttrDict(
    model=OneHotImagePriorBC,
))

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    skill_dim=data_spec.n_skills,
    n_rollout_steps=10,
    nz_enc=128,
    nz_mid=128,
    # n_processing_layers=5,
    # n_input_frames=1,
    img_enc_dim=128,
    prior_input_res=data_spec.res,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec