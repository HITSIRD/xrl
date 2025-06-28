import time

import numpy as np

from arm_controller.src.replay.replay import Replay
from arm_controller.src.utils.utils import string_to_float_array


def process_skill_data(suboperation_node, arm, hand=None):
    sub_type = suboperation_node.get('type')
    suboperations = suboperation_node.get('children')
    for suboperation in suboperations:
        processed_suboperation = process_suboperation_data(suboperation, arm, hand)
        time.sleep(0.5)
    print("Skill data processed")
    return True


# 处理子操作数据
def process_suboperation_data(suboperation_node, arm, hand=None):
    sub_type = suboperation_node.get('type')
    params = suboperation_node.get('params')
    import home.views
    home.views.skill_label = suboperation_node.get('description')

    # 根据不同的类型，写逻辑分支
    if sub_type == 'pose':
        pos_string = params['pose']
        result = string_to_float_array(pos_string)
        print(f"Processing suboperation {sub_type}: {params}")
        return arm.move_to_joint_position(np.array(result))

    elif sub_type == 'trajectory':
        path = params['path']
        file_name = params['file_name']
        # 合并完整路径
        full_path = f"{path}/{file_name}.h5"

        try:
            # 实例化 Replay 并执行重播
            replay = Replay(arm)
            replay.replay_trajectory(path=full_path)
            return True
        except Exception as e:
            print(f"轨迹重播失败: {e}")
            return False

    elif sub_type == 'gripper':
        width = float(params['state'])
        speed = 0.03
        force = 10.0
        print(f"Grasping to width{width} speed{speed} force{force}")
        return hand.grasp(width, speed, force)
    else:
        return {
            'type': sub_type,
            'description': '未知的子操作类型',
            'params': params
        }
