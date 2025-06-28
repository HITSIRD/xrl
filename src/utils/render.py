import re

import numpy as np
from dm_control import mujoco
from mujoco import mj_id2name
import matplotlib.pyplot as plt

# keywords = ['panda', 'knob 2', 'knob 4', 'lightswitch', 'slidelink', 'hingerightdoor', 'hingeright', 'micro', 'kettle']
keywords = ['knob 2', 'knob 4', 'lightswitch', 'slidelink', 'hingerightdoor', 'micro', 'kettle']

def render_mujoco_object_masks(env, state, object_names=keywords, img_size=256):
    """
    使用 MuJoCo segmentation 渲染每个目标物体的掩码图

    参数:
        env: MuJoCo 环境（如 D4RL Kitchen）
        object_names: 物体名称列表（如 ['microwave', 'kettle']
        state_dict: dict，包含环境状态（qpos, qvel）
        img_width, img_height: 渲染图像的尺寸

    返回:
        masks: list of (H, W) 布尔 numpy array，每个为一个物体的掩码
        names: list of str，物体名称列表，对应每个掩码
    """
    # --- 1. 设置环境状态 ---
    env._env.sim.set_state(np.concatenate([state[:30], np.zeros(29)]))
    env._env.sim.forward()
    # img = env._env.render("rgb_array", h=256, w=256)

    # === 2. 渲染 segmentation 图像 ===
    seg_image = env._env.render(mode='rgb_array', segmentation=True)
    # plt.imshow(seg_image[..., 0])
    # plt.show()

    masks = extract_segmentation_objects(seg_image, env._env.sim.model.ptr)
    merged = merge_masks_by_keywords(masks, object_names)

    # for name in merged.keys():
    #     print(f"{name}: mask sum = {merged[name].sum()}")
    #     plt.imshow(merged[name])
    #     plt.title(f'{name}')
    #     plt.show()

    return merged


def extract_segmentation_objects(seg_image, model):
    """
    从 segmentation 图中提取每个物体的名称和合并掩码。

    参数:
        seg_image: 渲染结果，形状 (H, W, 2)
        model: MuJoCo MjModel 对象

    返回:
        named_masks: dict，{name: mask (H, W)}，其中 mask 为 uint8 类型
    """
    obj_pairs = np.unique(seg_image.reshape(-1, 2), axis=0)
    named_masks = {}

    for obj_id, obj_type in obj_pairs:
        if obj_id == -1:
            continue  # 跳过背景

        obj_type = int(obj_type)
        obj_id = int(obj_id)

        # 获取名称
        name = mj_id2name(model, obj_type, obj_id)

        # 尝试获取更语义化的名称
        if name is None and obj_type == mujoco.mjtObj.mjOBJ_GEOM:
            body_id = model.geom_bodyid[obj_id]
            name = mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if name is None:
            name = f"unnamed_{obj_type}_{obj_id}"

        # 创建当前对象的掩码
        current_mask = ((seg_image[:, :, 0] == obj_id) &
                        (seg_image[:, :, 1] == obj_type)).astype(np.uint8)

        # 合并掩码
        if name in named_masks:
            named_masks[name] |= current_mask
        else:
            named_masks[name] = current_mask

    return named_masks


def merge_masks_by_keywords(named_masks, keywords):
    """
    根据关键词列表合并多个部分 mask，仅保留匹配上的。

    参数:
        named_masks: dict[str, np.ndarray]，name -> mask 映射
        keywords: list[str]，如 ['micro', 'panda']

    返回:
        merged_masks: dict[str, np.ndarray]，关键词 -> 合并后的 mask
    """
    # 初始化每个 keyword 的空 mask（尺寸来自任意已有 mask）
    sample_shape = next(iter(named_masks.values())).shape
    merged_masks = {key: np.zeros(sample_shape, dtype=np.uint8) for key in keywords}

    # 合并匹配到的 mask
    for name, mask in named_masks.items():
        for key in keywords:
            if re.search(key, name, re.IGNORECASE):
                merged_masks[key] |= mask
                break  # 避免重复归属

    # 丢弃没有合并出任何非零像素的项
    # merged_masks = {k: v for k, v in merged_masks.items() if v.any()}

    return merged_masks
