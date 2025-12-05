from src.data.real_kitchen.real_kitchen_dataloader import RealKitchenDataset
from src.utils.general import AttrDict

data_spec = AttrDict(
    # dataset_class=GlobalSplitVideoDataset,
    dataset_class=RealKitchenDataset,
    n_actions=7,
    state_dim=7,
    n_skills=5 + 1,
    # env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=256,
    # crop_rand_subseq=True,
    # max_seq_len = 100,
    max_seq_len=5,

    TASKS_DICT={
        'open_microwave': 0,
        'close_microwave': 1,
        'set_time': 2,
        'move_bread_to_microwave': 3,
        'move_bread_to_plate': 4,
        'end()': 5
    },

    skill_labels=['open_microwave', 'close_microwave', 'set_time', 'move_bread_to_microwave', 'move_bread_to_plate',
                  'end()'],

    skill_labels_ch=['开微波炉门', '关微波炉门', '设置时间', '面包移动到微波炉', '面包装盘', '结束'],

    objects=[["microwave",
              "bread",
              "switch",
              "plate"],

             ["microwave",
              "bread",
              "switch",
              "plate"],

             ["microwave",
              "bread",
              "switch",
              "plate"],

             ["microwave",
              "switch",
              "plate"],

             ["microwave",
              "switch",
              "plate"],

             ["microwave",
              "bread",
              "switch",
              "plate"],

             ["microwave",
              "bread",
              "switch",
              "plate"],
             ],

    boxes=[[[120, 8, 208, 102],
            [55, 140, 80, 160],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],

           [[120, 8, 208, 102],
            [55, 140, 80, 160],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],

           [[120, 8, 208, 102],
            [124, 49, 150, 71],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],

           [[120, 8, 208, 102],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],

           [[120, 8, 208, 102],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],

           [[120, 8, 208, 102],
            [124, 49, 150, 71],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],

           [[120, 8, 208, 102],
            [118, 170, 141, 203],
            [154, 73, 170, 88],
            [108, 164, 172, 218]],
           ],

    skill_obj_map={0: 'microwave',
                   1: 'bread',
                   2: 'microwave',
                   3: 'switch',
                   4: 'microwave',
                   5: 'bread',
                   6: 'microwave'},

    multi_skill_obj_map={0: ['microwave'],
                         1: ['microwave', 'bread'],
                         2: ['microwave'],
                         3: ['switch'],
                         4: ['microwave'],
                         5: ['bread', 'plate'],
                         6: ['microwave']},
)
