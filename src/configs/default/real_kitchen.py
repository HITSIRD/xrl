from src.data.real_kitchen.real_kitchen_dataloader import RealKitchenDataset
from src.utils.general import AttrDict

data_spec = AttrDict(
    # dataset_class=GlobalSplitVideoDataset,
    dataset_class=RealKitchenDataset,
    n_actions=7,
    state_dim=60,
    n_skills=5,
    # env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=128,
    # crop_rand_subseq=True,
    # max_seq_len = 100,
)
