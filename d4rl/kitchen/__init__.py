from .kitchen_envs import KitchenMicrowaveKettleLightSliderV0, KitchenMicrowaveKettleBottomBurnerLightV0, \
    KitchenMicrowaveLightSliderHingeV0, KitchenKettleBottomBurnerTopBurnerSliderV0, KitchenAllTasksV0
from gym.envs.registration import register

# Smaller dataset with only positive demonstrations.
register(
    id='kitchen-complete-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/mini_kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. A subset of the demonstrations
# solve the task.
register(
    id='kitchen-partial-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleLightSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_light_slider-v0.hdf5'
    }
)

# Whole dataset with undirected demonstrations. No demonstration completely
# solves the task, but each demonstration partially solves different
# components of the task.
register(
    id='kitchen-kbts-v0',
    entry_point='d4rl.kitchen:KitchenKettleBottomBurnerTopBurnerSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

register(
    id='kitchen-mlsh-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveLightSliderHingeV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

register(
    id='kitchen-mkbl-v0',
    entry_point='d4rl.kitchen:KitchenMicrowaveKettleBottomBurnerLightV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 4.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)

register(
    id='kitchen-all-v0',
    entry_point='d4rl.kitchen:KitchenAllTasksV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 7.0,
        'dataset_url': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/kitchen/kitchen_microwave_kettle_bottomburner_light-v0.hdf5'
    }
)
