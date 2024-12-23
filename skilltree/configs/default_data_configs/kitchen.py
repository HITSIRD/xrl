from skilltree.components.data_loader import GlobalSplitVideoDataset
from skilltree.utils.general_utils import AttrDict
from skilltree.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset, D4RLImageSequenceSplitDataset

data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    # dataset_class=D4RLSequenceSplitDataset,
    n_actions=9,
    state_dim=60,
    env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=200,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
