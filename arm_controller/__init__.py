from gym.envs.registration import register

register(
    id='RealKitchen-v0',
    entry_point='arm_controller.env.real_kitchen:RealRobotSkillEnv',
)