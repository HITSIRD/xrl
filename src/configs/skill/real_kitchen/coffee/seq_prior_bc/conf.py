from src.configs.default.coffee import data_spec
from src.configs.skill.real_kitchen.coffee.bc.conf import *
from src.data.real_kitchen.real_kitchen_dataloader import SequenceRealKitchenDataset
from src.models.bc import SequenceOneHotImagePriorBCModel
from src.utils.general import AttrDict

configuration.update(AttrDict(
    model=SequenceOneHotImagePriorBCModel,
))

data_spec.update(AttrDict(
    dataset_class=SequenceRealKitchenDataset,
))

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    skill_dim=data_spec.n_skills,
    rnn_hidden_dim=128,
    rnn_num_layers=1,

    nz_enc=128,
    nz_mid=128,
    # n_processing_layers=5,
    # n_input_frames=1,
    img_enc_dim=128,
    prior_input_res=data_spec.res,
    use_resnet=True
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
