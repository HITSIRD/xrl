from src.data.real_kitchen.real_kitchen_dataloader import RealKitchenDataset
from src.utils.general import AttrDict

data_spec = AttrDict(
    # dataset_class=GlobalSplitVideoDataset,
    dataset_class=RealKitchenDataset,
    n_actions=7,
    state_dim=7,
    n_skills=8 + 1,
    # env_name="kitchen-mkbl-v0",
    split=AttrDict(train=0.99, val=0.01, test=0.0),
    res=256,
    # crop_rand_subseq=True,
    # max_seq_len = 100,
    max_seq_len=5,

    TASKS_DICT={
        'move(funnel, table)': 0,
        'move(funnel, pot)': 1,
        'pour_preheat(gooseneck_kettle, funnel)': 2,
        'pour(gooseneck_kettle, funnel)': 3,
        'pour(pot, cup)': 4,
        'pour(pot, coffee_cup)': 5,
        'pour(coffee_powder, funnel)': 6,
        'pour(kettle, gooseneck_kettle)': 7,
        'end()': 8
    },

    skill_labels=['move(funnel, table)', 'move(funnel, pot)', 'pour_preheat(gooseneck_kettle, funnel)',
                  'pour(gooseneck_kettle, funnel)', 'pour(pot, cup)',
                  'pour(pot, coffee_cup)', 'pour(coffee_powder, funnel)', 'pour(kettle, gooseneck_kettle)', 'end()'],

    skill_labels_ch=['漏斗放桌上', '漏斗放咖啡壶上', '向咖啡壶中预热倒水', '向咖啡壶中倒水', '咖啡壶倒入杯中',
                     '咖啡壶倒入咖啡杯', '咖啡粉倒入漏斗', '水壶倒入手冲壶', '结束'],

    objects=[[
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],

             [
              "funnel",
              "pot",
              "kettle",
              "gooseneck_kettle",
              "cup",
              "coffee_cup",
              "coffee_powder"],
             ],

    boxes=[[
            [112, 54, 144, 89],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [112, 54, 144, 89],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [112, 54, 144, 89],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [135, 14, 167, 47],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]],

           [
            [112, 54, 144, 89],
            [132, 34, 173, 73],
            [112, 118, 145, 162],
            [182, 51, 221, 82],
            [113, 34, 128, 59],
            [102, 49, 123, 70],
            [97, 62, 113, 80]]
           ],

    skill_obj_map={0: 'funnel',
                   1: 'gooseneck_kettle',
                   2: 'funnel',
                   3: 'pot',
                   4: 'funnel',
                   5: 'coffee_powder',
                   6: 'kettle',
                   7: 'gooseneck_kettle',
                   8: 'gooseneck_kettle',
                   9: 'funnel',
                   10: 'pot'},

    multi_skill_obj_map={0: ['funnel', 'pot'],
                         1: ['gooseneck_kettle', 'funnel'],
                         2: ['funnel'],
                         3: ['pot', 'cup'],
                         4: ['funnel', 'pot'],
                         5: ['coffee_powder', 'funnel'],
                         6: ['kettle', 'gooseneck_kettle'],
                         7: ['gooseneck_kettle', 'funnel'],
                         8: ['gooseneck_kettle', 'funnel'],
                         9: ['funnel'],
                         10: ['pot', 'coffee_cup']},
)
