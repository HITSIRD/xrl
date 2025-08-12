from src.data.real_kitchen.real_kitchen_dataloader import RealKitchenDataset
from src.utils.general import AttrDict

data_spec = AttrDict(
    # dataset_class=GlobalSplitVideoDataset,
    dataset_class=RealKitchenDataset,
    n_actions=7,
    state_dim=7,
    n_skills=9,
    # env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=256,
    # crop_rand_subseq=True,
    # max_seq_len = 100,

    labels=['open_fridge', 'close_fridge', 'open_cab', 'close_cab', 'store_mango', 'store_lemon', 'store_orange',
            'store_cheezit', 'store_jello'],

    objects=[["fridge",
             "cab",
             "mango",
             "lemon",
             "orange",
             "cheezit",
             "jello"],

             ["fridge",
              "cab",
              "mango",
              "lemon",
              "orange",
              "cheezit",
              "jello"],

             ["fridge",
              "cab",
              "cheezit",
              "jello"],

             ["fridge",
              "cab"],

             ["fridge",
              "cab",
              "mango",
              "lemon",
              "orange",
              "cheezit",
              "jello"],

             ["fridge",
              "cab",
              "mango",
              "lemon",
              "orange",
              "cheezit",
              "jello"],

             ["fridge",
              "cab",
              "mango",
              "lemon",
              "orange",
              "cheezit",
              "jello"],

             ["fridge",
              "cab",
              "cheezit",
              "jello"],

             ["fridge",
              "cab",
              "jello"]
             ],

    boxes=[[[144, 48, 250, 174],
           [51, 15, 134, 130],
           [98, 148, 126, 178],
           [118, 180, 144, 210],
           [115, 118, 140, 145],
           [56, 126, 82, 156],
           [155, 182, 194, 248]],

           [[144, 48, 250, 174],
            [51, 15, 134, 130],
            [180, 117, 198, 139],
            [168, 105, 187, 128],
            [190, 108, 212, 128],
            [56, 126, 82, 156],
            [155, 182, 194, 248]],

            [[144, 48, 250, 174],
            [51, 15, 134, 130],
            [56, 126, 82, 156],
            [155, 182, 194, 248]],

           [[144, 48, 250, 174],
            [51, 15, 134, 130]],

           [[144, 48, 250, 174],
           [51, 15, 134, 130],
           [98, 148, 126, 178],
           [118, 180, 144, 210],
           [115, 118, 140, 145],
           [56, 126, 82, 156],
           [155, 182, 194, 248]],

           [[144, 48, 250, 174],
            [51, 15, 134, 130],
            [180, 117, 198, 139],
            [118, 180, 144, 210],
            [115, 118, 140, 145],
            [56, 126, 82, 156],
            [155, 182, 194, 248]],

           [[144, 48, 250, 174],
            [51, 15, 134, 130],
            [180, 117, 198, 139],
            [168, 105, 187, 128],
            [115, 118, 140, 145],
            [56, 126, 82, 156],
            [155, 182, 194, 248]],

           [[144, 48, 250, 174],
            [51, 15, 134, 130],
            [56, 126, 82, 156],
            [155, 182, 194, 248]],

           [[144, 48, 250, 174],
            [51, 15, 134, 130],
            [155, 182, 194, 248]]
           ],

    gt_skill_index=[0, 1, 2, 3, 4, 5, 6, 7, 8]
)
