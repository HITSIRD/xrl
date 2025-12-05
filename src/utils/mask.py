import numpy as np
import re
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt


def _to_bool_mask(mask):
    arr = np.array(mask)
    return arr > 0.5 if arr.dtype.kind == "f" else arr.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return bbox (ymin,xmin,ymax,xmax) for boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _bbox_distance(b1, b2) -> float:
    """Pixel distance between two bboxes (0 if overlap). b = (ymin,xmin,ymax,xmax)"""
    if b1 is None or b2 is None:
        return float('inf')
    y1min, x1min, y1max, x1max = b1
    y2min, x2min, y2max, x2max = b2
    dx = max(y2min - y1max, y1min - y2max, 0)
    dy = max(x2min - x1max, x1min - x2max, 0)
    # dx is y-distance, dy is x-distance
    return float(np.sqrt(dx * dx + dy * dy))


def assign_regions_from_dicts(mask_dicts):
    H, W = mask_dicts[0]["segmentation"].shape
    assigned = np.zeros((H, W), dtype=bool)  # 已分配区域
    assigned_masks = []

    # 按 score 从高到低排序
    sorted_idx = np.argsort([d["stability_score"] for d in mask_dicts])[::-1]

    for idx in sorted_idx:
        mask = mask_dicts[idx]["segmentation"].astype(bool)
        new_mask = np.logical_and(mask, ~assigned)
        assigned_masks.append({
            "segmentation": new_mask,
        })
        assigned = np.logical_or(assigned, new_mask)

    assigned_dicts = [None] * len(mask_dicts)
    for i, idx in enumerate(sorted_idx):
        assigned_dicts[idx] = assigned_masks[i]

    return assigned_dicts


def merge_mask(mask_dict: Dict[str, np.ndarray],
               unnamed_masks: List[Dict[str, np.ndarray]],
               min_mask_area=300,
               iou_exclude_thresh: float = 0.2) -> Dict[str, np.ndarray]:
    """合并已有 mask_dict 和未命名 masks（剔除与已有mask重叠过大的）"""
    merged = {k: v for k, v in mask_dict.items()}
    named_masks = list(merged.values())

    filtered_unnamed_masks = assign_regions_from_dicts(unnamed_masks)

    filtered_unnamed_masks = [
        mask for mask in filtered_unnamed_masks
        if mask["segmentation"].sum() >= min_mask_area
    ]

    # if filtered_unnamed_masks:
    #     fig, axes = plt.subplots(1, len(filtered_unnamed_masks), figsize=(35, 1))
    #     if len(filtered_unnamed_masks) == 1:
    #         axes = [axes]
    #     for i, mask in enumerate(filtered_unnamed_masks):
    #         axes[i].imshow(mask["segmentation"])
    #         # axes[i].set_title(f"Mask {i+1}")
    #         axes[i].axis('off')
    #     plt.suptitle("Filtered Unnamed Masks")
    #     plt.show()

    # 找到已有 object_N 最大编号
    existing_ids = [int(m.group(1)) for k in merged.keys()
                    if (m := re.match(r"^object_(\d+)$", k))]
    next_id = max(existing_ids, default=0) + 1

    for cand in filtered_unnamed_masks:
        m = cand["segmentation"]
        # 跳过与已有mask重合度过高的

        if any(_iou(m, nm) >= iou_exclude_thresh for nm in named_masks):
            continue
        merged[f"object_{next_id}"] = m
        named_masks.append(m)
        next_id += 1

    return merged
