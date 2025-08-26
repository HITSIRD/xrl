import random
import cv2
from PIL import Image
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import h5py
import os
import groundingdino.datasets.transforms as T

import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from transformers import CLIPProcessor, CLIPModel
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from depth_anything_v2.dpt import DepthAnythingV2

sam2_path = os.path.dirname(sam2.__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 初始化SAM模型
def initialize_sam(sam_checkpoint="/home/wenyongyan/下载/sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


sam = sam_model_registry["vit_h"](checkpoint="/home/wenyongyan/下载/sam_vit_h_4b8939.pth")
sam.to(device)

checkpoint = "/home/wenyongyan/下载/sam2.1_hiera_large.pt"
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

groundingdino_model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py",
                                 "groundingdino/weights/groundingdino_swinb_cogcoor.pth")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(device).eval()


# 初始化CLIP模型
def initialize_clip(model_name="openai/clip-vit-base-patch32", device="cuda"):
    model_path = "clip_model"  # 你的本地模型存放路径
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor


def get_image_embedding(image_path, clip_model, preprocess, device):
    """提取单张图像的 CLIP 特征"""
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(img)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding


def get_average_embedding(image_paths, clip_model, preprocess, device):
    """提取多张参考图的平均特征"""
    embeddings = [get_image_embedding(p, clip_model, preprocess, device) for p in image_paths]
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    avg_embedding /= avg_embedding.norm(dim=-1, keepdim=True)
    return avg_embedding


def detect_candidates_with_dino(model, image_path, caption="object", box_threshold=0.1, text_threshold=0.1):
    """用 GroundingDINO 检测候选框"""
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    h, w, _ = image_source.shape
    boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
    boxes = boxes.cpu().numpy().astype(int)

    return image_source, boxes


def get_depth(img):
    return model.infer_image(img)


def box_inside(inner, outer):
    return (inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[2] <= outer[2] and inner[3] <= outer[3])


def remove_foreground_overlap(boxes, masks):
    n = len(boxes)
    masks_visible = [m.copy() for m in masks]  # 拷贝避免修改原 mask

    # 遍历每对 box
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # 如果 i 在 j 内部，i 是前景，j 是背景
            if box_inside(boxes[i], boxes[j]):
                masks_visible[j] = masks_visible[j] & (~masks[i])

    return masks_visible


def groundingdino_box_prompt(image, save_dir=None):
    # TEXT_PROMPT = "kettle . microwave ."

    global frame
    TEXT_PROMPT = "object"
    BOX_TRESHOLD = 0.2
    TEXT_TRESHOLD = 0.2

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_source = Image.fromarray(image).convert("RGB")
    plt.imshow(get_depth(image))
    plt.show()

    # image_source = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image_transformed,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    print(boxes)
    print(logits)
    print(phrases)

    if len(boxes) > 0:
        # box = boxes[0]
        # logits = logits[0]

        # print(boxes)
        # print(logits)

        annotated_frame = annotate(image_source=np.asarray(image_source), boxes=boxes,
                                   logits=logits,
                                   phrases=phrases)
        # plt.imsave(f'{save_dir}/{frame}.png', cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB
        plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB
        plt.show()

        # box = _process_box(boxes[0])
        # box = box * 256
        return box.cpu().numpy().tolist()

    annotated_frame = annotate(image_source=np.asarray(image_source), boxes=boxes, logits=logits, phrases=phrases)
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB

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
def generate_masks_with_sam(image, labels=None, boxes=None, combined=False, save_dir=None):
    """
    使用SAM生成图像中的所有分割掩码
    """
    # mask_generator = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.95, box_nms_thresh=0.1, crop_nms_thresh=0.5,
    #                                            crop_n_layers=1, min_mask_region_area=50)
    # masks = mask_generator.generate(image_obs)

    predictor.set_image(image)

    if boxes is None:
        # masks, _, _ = predictor.predict()

        mask_generator = SamAutomaticMaskGenerator(model=sam)
        masks = mask_generator.generate(image)

        if combined:
            return combine_masks(masks)
        return masks
    else:
        masks = []
        masks_dict = {}
        boxes = boxes.copy()

        for i, box in enumerate(boxes):
            # 将边界框转换为 [x_min, y_min, x_max, y_max]
            bbox_coords = np.array([box])  # 这里bbox_coords是一个二维数组
            # 使用边界框生成分割掩码
            mask, _, _ = predictor.predict(box=bbox_coords, multimask_output=False)
            mask = mask[0].astype(np.uint8)
            masks.append(mask)

        remove_foreground_overlap(boxes, masks)
        for i, box in enumerate(boxes):
            masks_dict[labels[i]] = masks[i]
            # plt.title(labels[i])
            # plt.imshow(masks[i])
            # plt.show()

        return masks_dict


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


def combine_masks(mask_list):
    h, w = mask_list[0]['segmentation'].shape
    combined = np.zeros((h, w), dtype=np.uint8)  # 初始化为0，表示背景

    for i, mask in enumerate(mask_list, start=1):
        # 用 i 表示第 i 个 mask 区域
        combined[mask['segmentation']] = i

    return combined


def visualize_masks(masks, image):
    mask_overlay = np.zeros_like(image, dtype=np.uint8)

    # 为每个 mask 分配一种随机颜色
    for i, mask in enumerate(masks):
        color = [random.randint(0, 255) for _ in range(3)]  # 随机颜色
        mask_overlay[mask['segmentation'] > 0] = color  # 应用颜色到 mask 区域

    # 将 mask 叠加到原图上
    blended_image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

    # 使用 matplotlib 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))  # 转换 BGR 到 RGB
    plt.axis("off")
    plt.title("Mask Visualization")
    plt.show()


def main(image, labels):
    # 初始化模型
    # sam_predictor = initialize_sam(device=device)
    # clip_model, clip_processor = initialize_clip(device=device)

    # groundingdino_box_prompt(image)

    # 使用SAM生成分割掩码
    # masks = generate_masks_with_sam(image, labels)

    masks = generate_masks_with_sam(image)

    visualize_masks(masks, image)

    # valid_masks = filter_shadow_masks(masks, shadow_suppression(image_obs))

    # 使用CLIP对每个分割区域进行分类
    # results = classify_with_clip(clip_model, clip_processor, image_obs, masks, candidate_labels)

    # 可视化结果
    # visualize_results(image_obs, results)


# 运行主程序
if __name__ == "__main__":
    res = 256

    file = 'src/experiments/hrl/real_kitchen/fruits_snacks/multi_steps_prior_bc/top50/sample_rollout_0.h5'

    with h5py.File(file, 'r') as dataset:
        print(dataset.keys())
        # print(dataset['traj']['states'])
        image_obs = dataset['states'][3]

    image_obs = image_obs[7:].reshape(3, res, res) * 255 / 2 + 128
    image_obs = image_obs.astype(np.uint8)
    image_obs = np.transpose(image_obs, [1, 2, 0])

    # 设置候选类别标签
    # candidate_labels = ["microwave", "kettle", "rotary switch", "slide cabinet", "hinge cabinet",
    #                     "burner",
    #                     "robot arm", "handle"]

    candidate_labels = ["Fridge"]
    labels = ['fridge']

    # 运行主函数
    main(image_obs, labels)
