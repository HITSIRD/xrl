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

    labels=["bottom burner switch",
            "top burner switch",
            "light switch",
            "slide cabinet",
            "hinge cabinet",
            "microwave",
            "kettle"],

    boxes=[[98, 59, 115, 74],
           [98, 46, 115, 59],
           [76, 51, 96, 72],
           [50, 0, 110, 50],
           [110, 0, 200, 50],
           [0, 55, 75, 130]],

    gt_skill_index = [6, 0, 1, 3],
)
