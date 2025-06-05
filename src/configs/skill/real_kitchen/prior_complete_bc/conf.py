from src.configs.default.real_kitchen import data_spec
from src.configs.skill.real_kitchen.bc.conf import *
from src.models.bc import OneHotImagePriorCompleteBCModel
from src.utils.general import AttrDict

configuration.update(AttrDict(
    model=OneHotImagePriorCompleteBCModel,
))

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    skill_dim=data_spec.n_skills,
    nz_enc=128,
    nz_mid=128,
    img_enc_dim=128,
    prior_input_res=data_spec.res,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec