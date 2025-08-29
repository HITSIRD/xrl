from gym.envs.registration import register

register(
    id='FridgeMangoCabJello-v0',
    entry_point='arm_controller.env.real_kitchen:FridgeMangoCabJello',
)

register(
    id='FruitsSnacks-v0',
    entry_point='arm_controller.env.real_kitchen:FruitsSnacks',
)

register(
    id='HeatBread-v0',
    entry_point='arm_controller.env.real_kitchen:HeatBread',
)