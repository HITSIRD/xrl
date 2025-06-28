import random

import cv2
import gym
import d4rl
import mmcv
import torch
from PIL import Image

from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from sympy.logic.inference import valid
import groundingdino.datasets.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#
#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)
#
# file = 'data/kitchen/kitchen-mixed-v0/kitchen-mixed-v0_0.h5'
#
# with h5py.File(file, 'r') as dataset:
#     # print(dataset['traj'].keys())
#     # print(dataset['traj']['states'])
#     image_obs = dataset['traj']['observations'][150]
#
# sam = sam_model_registry["vit_h"](checkpoint="/home/wenyongyan/下载/sam_vit_h_4b8939.pth")
# sam.to(device)
#
# mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=100)
# masks = mask_generator.generate(image_obs)
#
# # predictor = SamPredictor(sam)
# # predictor.set_image(image_obs)  # 设置输入图像
# #
# # input_box = [0, 0, 200, 200]
# #
# # masks, scores, logits = predictor.predict(
# #     box=torch.tensor(input_box).to(torch.float32).unsqueeze(0),
# #     multimask_output=True)
#
# print(len(masks))
# # print(masks)
#
# # plt.figure(figsize=(10, 10))
# plt.imshow(image_obs)
# show_anns(masks)
# plt.title("Segment Anything Model")
# plt.axis("off")
# plt.show()


# from mmdet.apis import init_detector, inference_detector
# from mmdet.registry import VISUALIZERS
# import cv2
# import mmcv
#
# # 配置文件路径和模型权重文件路径
# config_file = '/home/wenyongyan/下载/mmdetection/mask-rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = '/home/wenyongyan/下载/mmdetection/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
#
# # 初始化模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
#
# # 测试图像
# file = 'skilltree/data/kitchen/kitchen-mixed-v0/kitchen-mixed-v0_0.h5'
#
# with h5py.File(file, 'r') as dataset:
#     image_obs = dataset['traj']['observations'][199]
# result = inference_detector(model, image_obs)
#
# visualizer = VISUALIZERS.build(model.cfg.visualizer)
# visualizer.dataset_meta = model.dataset_meta
#
# visualizer.add_datasample(
#     name='result',
#     image=image_obs,
#     data_sample=result,
#     draw_gt=False,
#     pred_score_thr=0.1,
#     show=False)
#
# img = visualizer.get_image()
# plt.imshow(img)
# plt.show()
#
# plt.savefig('img.png')
#


## deepseek-r1 pipeline

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from transformers import CLIPProcessor, CLIPModel
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_path = os.path.dirname(sam2.__file__)

# 初始化SAM模型
# def initialize_sam(sam_checkpoint="/home/wenyongyan/下载/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#     sam.to(device=device)
#     return SamPredictor(sam)

# sam = sam_model_registry["vit_h"](checkpoint="/home/wenyongyan/下载/sam_vit_h_4b8939.pth")
# sam.to(device)

checkpoint = "/home/wenyongyan/下载/sam2.1_hiera_large.pt"
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

groundingdino_model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py",
                                 "groundingdino/weights/groundingdino_swinb_cogcoor.pth")
frame = 0


# 初始化CLIP模型
def initialize_clip(model_name="openai/clip-vit-base-patch32", device="cuda"):
    model_path = "clip_model"  # 你的本地模型存放路径
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def groundingdino_box_prompt(image, save_dir):
    # TEXT_PROMPT = "kettle . microwave ."

    global frame
    TEXT_PROMPT = "metallic water kettle"
    BOX_TRESHOLD = 0.15
    TEXT_TRESHOLD = 0.15

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_source = Image.fromarray(image).convert("RGB")
    # image_source = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # print(boxes)
    # print(logits)
    # print(phrases)

    frame += 1

    if len(boxes) > 0:
        valid_indices = [i for i, box in enumerate(boxes) if box[2] <= 0.3 and box[3] <= 0.3]
        print(valid_indices)
        if len(valid_indices) > 0:
            box = boxes[valid_indices[0]]
            logits = logits[valid_indices[0]]

            # print(boxes)
            # print(logits)

            annotated_frame = annotate(image_source=np.asarray(image_source), boxes=box.unsqueeze(0),
                                       logits=logits.unsqueeze(0),
                                       phrases=phrases)
            plt.imsave(f'{save_dir}/{frame}.png', cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB

            # box = _process_box(boxes[0])
            box = box * 200
            return box.cpu().numpy().tolist()

    # annotated_frame = annotate(image_source=np.asarray(image_source), boxes=boxes, logits=logits, phrases=phrases)
    # plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB

    return None


def _process_box(box):
    x, y = box[0], box[1]
    w, h = box[2], box[3]
    box[0] = x - w / 2
    box[1] = y - h / 2
    box[2] = x + w / 2
    box[3] = y + h / 2
    return box


# 使用SAM生成分割掩码
def generate_masks_with_sam(image, boxes=None, save_dir=None):
    """
    使用SAM生成图像中的所有分割掩码
    """
    # plt.imsave(f'original_image_{frame}.png', image)  # 转换 BGR 到 RGB
    # plt.imsave(f'upscale_image_{frame}.png', image_obs)  # 转换 BGR 到 RGB

    # predictor.set_image(image)
    # masks, _, _ = predictor.predict()

    # sam = sam_model_registry["vit_h"](checkpoint="/home/wenyongyan/下载/sam_vit_h_4b8939.pth")
    # sam.to(device)

    # mask_generator = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.95, box_nms_thresh=0.1, crop_nms_thresh=0.5,
    #                                            crop_n_layers=1, min_mask_region_area=50)
    # masks = mask_generator.generate(image_obs)

    predictor.set_image(image)
    # masks, _, _ = predictor.predict(point_coords=np.array([[60, 105], [37, 40], [60, 40], [130, 40], [82, 60],
    #                                                        [110, 52], [110, 63], [100, 130]]),
    #                                 point_labels=np.array([1, 1, 1, 1, 1, 1, 1, 1]), multimask_output=True)

    # masks, _, _ = predictor.predict(box=np.array([0, 50, 75, 125]), multimask_output=False)

    # boxes = [[0, 50, 75, 150], [50, 0, 100, 50], [125, 0, 200, 50], [75, 50, 85, 70], [85, 50, 95, 70],
    #          [95, 60, 110, 75]]
    masks = []
    boxes = boxes.copy()
    size = image.shape[0]

    box = groundingdino_box_prompt(image, save_dir)
    if box is None:
        box = [80, 110, 120, 150]
    boxes.append(box)

    print(boxes)

    for box in boxes:
        # 将边界框转换为 [x_min, y_min, x_max, y_max]
        bbox_coords = np.array([box]) * size / 200  # 这里bbox_coords是一个二维数组
        # 使用边界框生成分割掩码
        mask, _, _ = predictor.predict(box=bbox_coords, multimask_output=False)
        masks.append(cv2.resize(mask[0].astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))

    return masks


# 使用CLIP对分割区域进行分类
def classify_with_clip(model, processor, image, masks, candidate_labels):
    """
    使用CLIP对每个分割区域进行分类
    """
    results = []
    for mask in masks:
        # 裁剪目标区域
        # mask = mask['segmentation']
        masked_image = apply_mask_to_image(image, mask.squeeze(0))

        # CLIP分类
        inputs = processor(text=candidate_labels, images=masked_image,
                           return_tensors="pt", padding=True).to(model.device)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        # 获取分类结果
        label = candidate_labels[probs.argmax()]
        confidence = probs.max().item()

        results.append({
            'mask': mask,
            'label': label,
            'confidence': confidence
        })

    print(f'object: {len(results)}')
    # print(results)
    return results


# 将掩码应用到图像上
def apply_mask_to_image(image, mask):
    """
    将二值掩码应用到图像上，裁剪出目标区域
    """
    masked_image = np.zeros_like(image)
    masked_image[mask > 0] = image[mask > 0]

    # 获取掩码的边界框
    y, x = np.where(mask)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 裁剪目标区域
    cropped_image = masked_image[y_min:y_max, x_min:x_max]
    return cropped_image


# 可视化结果
def visualize_results(image, results):
    """
    可视化分割和分类结果，使用不同颜色的 mask 进行显示
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    overlay = np.zeros_like(image, dtype=np.uint8)  # 创建叠加层
    color_map = {}  # 存储类别对应的颜色

    for result in results:
        mask = result['mask']
        label = result['label']
        confidence = result['confidence']

        # 为每个类别分配唯一颜色
        if label not in color_map:
            color_map[label] = [random.randint(100, 255) for _ in range(3)]  # 生成随机颜色

        # 叠加 mask
        for c in range(3):  # RGB 三通道叠加
            overlay[:, :, c] = np.where(mask > 0, color_map[label][c], overlay[:, :, c])

        # 计算 mask 中心点
        M = cv2.moments(mask.squeeze(0).astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            plt.text(cx, cy, f"{label} ({confidence:.2f})",
                     color='white', fontsize=10, backgroundcolor='black')

        # 将 overlay 叠加到原图上，使用透明度 0.5
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        plt.imshow(blended)

    # plt.axis('off')
    plt.show()


def shadow_suppression(img):
    """
    三阶段阴影抑制处理
    """
    # 阶段1: 自适应直方图均衡化
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 阶段2: 饱和度增强
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)

    # 阶段3: 阴影区域弱化
    _, light_mask = cv2.threshold(l_eq, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shadow_mask = cv2.bitwise_not(light_mask)
    hsv[:, :, 2] = np.where(shadow_mask > 0,
                            hsv[:, :, 2] * 0.8,
                            hsv[:, :, 2])

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def is_shadow_region(mask, image_hsv,
                     brightness_thresh=60,
                     saturation_thresh=30):
    """
    基于HSV特征的阴影区域检测
    """
    # 提取掩码区域的HSV特征
    mask_area = image_hsv[mask['segmentation'] > 0]
    avg_brightness = np.mean(mask_area[:, 2])
    avg_saturation = np.mean(mask_area[:, 1])

    # 阴影判断规则
    if avg_brightness < brightness_thresh and \
            avg_saturation < saturation_thresh:
        return True
    return False


def visualize_masks(masks, image):
    mask_overlay = np.zeros_like(image, dtype=np.uint8)

    # 为每个 mask 分配一种随机颜色
    for i, mask in enumerate(masks):
        color = [random.randint(0, 255) for _ in range(3)]  # 随机颜色
        mask_overlay[mask[0] > 0] = color  # 应用颜色到 mask 区域

    # 将 mask 叠加到原图上
    blended_image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

    # 使用 matplotlib 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB
    plt.axis("off")
    plt.title("Mask Visualization")
    plt.show()


def filter_shadow_masks(masks, image):
    """
    过滤阴影掩码
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    valid_masks = []
    for mask in masks:
        if not is_shadow_region(mask, hsv):
            valid_masks.append(mask)
    return valid_masks


def main(image):
    # 初始化模型
    # sam_predictor = initialize_sam(device=device)
    # clip_model, clip_processor = initialize_clip(device=device)

    groundingdino_box_prompt(image)

    # 使用SAM生成分割掩码
    masks = generate_masks_with_sam(image)

    visualize_masks(masks, image)

    # valid_masks = filter_shadow_masks(masks, shadow_suppression(image_obs))

    # 使用CLIP对每个分割区域进行分类
    # results = classify_with_clip(clip_model, clip_processor, image_obs, masks, candidate_labels)

    # 可视化结果
    # visualize_results(image_obs, results)


# 运行主程序
if __name__ == "__main__":
    res = 512
    env = gym.make('kitchen-all-v0')
    dataset = env.get_dataset()
    observations = dataset['observations']

    # file = 'skilltree/data/kitchen/kitchen-mixed-v0/kitchen-mixed-v0_101.h5'
    #
    # with h5py.File(file, 'r') as dataset:
    #     # print(dataset['traj'].keys())
    #     # print(dataset['traj']['states'])
    #     image_obs = dataset['traj']['observations'][150]

    obs = observations[10000]
    obs_dict = {"qp": obs[:9], "obj_qp": obs[9:30]}
    reward_dict, score, completions = env.env._get_reward_n_score(obs_dict)

    env.env.sim.set_state(np.concatenate([obs[:30], np.zeros(29)]))
    env.env.sim.forward()
    image_obs = np.array(env.env.render("rgb_array", h=res, w=res))

    # 设置候选类别标签
    # candidate_labels = ["microwave", "kettle", "rotary switch", "slide cabinet", "hinge cabinet",
    #                     "burner",
    #                     "robot arm", "handle"]

    candidate_labels = ["A black kitchen stove with circular burners, located in a kitchen setting.",
                        # "A robotic arm with white metallic texture",
                        "A black microwave oven.",
                        "A dark-colored kitchen cabinet, dark-colored.",
                        "Stove dials, used to adjust the burners.",
                        "A toggle switch.",
                        "A kettle with brown handle."]

    # 运行主函数
    main(image_obs)
