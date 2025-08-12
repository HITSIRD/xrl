from src.models.bc import OneHotImageBCModel
from src.utils.general import AttrDict
import os

configuration = AttrDict(
    model=OneHotImageBCModel,
    data_dir=os.path.join(os.environ['DATA_DIR'], 'real_kitchen/fruits-snacks-50-v0'),
    epoch_cycles_train=20,
    num_epochs=10,

    img_res=256,
    num_skills=9,
    action_dim=7,
    lr=0.0003,
    batch_size=256,
    )