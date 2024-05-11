from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset, GlobalSplitDataset
from spirl.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset



data_spec = AttrDict(
    dataset_class=GlobalSplitDataset,
    n_actions=2,
    state_dim=4,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    res=32,
    crop_rand_subseq=True,
)
# data_spec.max_seq_len = 300
data_spec.max_seq_len = 2000
