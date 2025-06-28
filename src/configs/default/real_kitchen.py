from src.data.real_kitchen.real_kitchen_dataloader import RealKitchenDataset
from src.utils.general import AttrDict

data_spec = AttrDict(
    # dataset_class=GlobalSplitVideoDataset,
    dataset_class=RealKitchenDataset,
    n_actions=7,
    state_dim=7,
    n_skills=5,
    # env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=256,
    # crop_rand_subseq=True,
    # max_seq_len = 100,

    labels=["microwave",
            "slide cabinet",
            "hinge cabinet",
            "light switch",
            "top burner switch",
            "bottom burner switch",
            "kettle"],
    boxes=[[0, 50, 75, 150], [50, 0, 100, 50], [125, 0, 200, 50], [75, 50, 85, 70], [85, 50, 95, 70],
           [95, 60, 110, 75]]
)
