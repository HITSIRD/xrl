from src.data.kitchen.kitchen_dataloader import KitchenDataset
from src.utils.general import AttrDict

data_spec = AttrDict(
    # dataset_class=GlobalSplitVideoDataset,
    dataset_class=KitchenDataset,
    n_actions=9,
    state_dim=60,
    n_skills=7,
    env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=256,
    # crop_rand_subseq=True,
    # max_seq_len = 100,
)
