import copy

import numpy as np
import torch
from captum.attr import IntegratedGradients, GradientShap, GuidedGradCam, LayerGradCam, DeepLift, DeepLiftShap, LRP, \
    Occlusion, GuidedBackprop, InputXGradient, Lime, FeaturePermutation, FeatureAblation, Deconvolution, \
    ShapleyValueSampling
from captum.attr._utils.lrp_rules import EpsilonRule

from src.instance_seg_test import generate_masks_with_sam
from src.utils.dist import kl_categorical
from src.utils.image import gaussian_blur_perturb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_saliency(exp_method, policy, processed_img, skill_idx, original_img=None):
    if exp_method == 'ig':
        return compute_integrated_gradient_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'gradient':
        return compute_gradient_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'gradient_shap':
        return compute_gradient_shap(policy, processed_img, skill_idx)
    elif exp_method == 'gaussian_perturbation':
        return compute_perturbation_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'guided_backprop':
        return compute_guided_backprop_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'input_x_gradient':
        return compute_input_x_gradient_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'grad_cam':
        return compute_gradCAM_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'guided_grad_cam':
        return compute_gradCAM_saliency(policy, processed_img, skill_idx, layer=False)
    elif exp_method == 'deep_lift':
        return compute_deeplift_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'deep_shap':
        return compute_deepSHAP_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'lrp':
        return compute_lrp_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'occlusion':
        return compute_occlusion_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'lime':
        return compute_lime_saliency(policy, processed_img, skill_idx,
                                     generate_masks_with_sam(original_img, combined=True))
    elif exp_method == 'feature_ablation':
        return compute_feature_ablation_saliency(policy, processed_img, skill_idx,
                                                 generate_masks_with_sam(original_img, combined=True))
    elif exp_method == 'deconvolution':
        return compute_deconvolution_saliency(policy, processed_img, skill_idx)
    elif exp_method == 'svs':
        return compute_shapley_value_sampling_saliency(policy, processed_img, skill_idx,
                                                       generate_masks_with_sam(original_img, combined=True))
    else:
        raise Exception(f'unsupported explanation method {exp_method}')


def compute_lrp_saliency(policy, img, skill_idx):
    def remove_logsoftmax(module):
        """
        递归地删除模型中的所有 nn.LogSoftmax 层，
        如果嵌套在 nn.Sequential 中，则直接从序列中移除
        """
        for name, child in list(module.named_children()):
            # 如果是 Sequential，则构造新的子模块列表
            if isinstance(child, torch.nn.Sequential):
                new_layers = []
                for sub_child in child.children():
                    if not isinstance(sub_child, torch.nn.LogSoftmax):
                        new_layers.append(sub_child)
                # 重新赋值
                setattr(module, name, torch.nn.Sequential(*new_layers))
            elif isinstance(child, torch.nn.LogSoftmax):
                # 如果不是 Sequential 中的，直接移除（替换为 nn.Identity 或删除）
                delattr(module, name)
            else:
                # 递归处理子模块
                remove_logsoftmax(child)

    policy = copy.copy(policy)
    remove_logsoftmax(policy.prior_head)
    lrp = LRP(policy)
    attr = lrp.attribute(img, target=skill_idx.item()).detach().cpu().numpy()
    saliency_map = np.linalg.norm(attr, ord=2, axis=1).squeeze()

    return saliency_map


def compute_occlusion_saliency(policy, img, skill_idx):
    occ = Occlusion(policy)
    attr = occ.attribute(img, target=skill_idx.item(), sliding_window_shapes=(1, 10, 10), perturbations_per_eval=256,
                         show_progress=True).detach().cpu().numpy()

    saliency_map = np.linalg.norm(attr, ord=2, axis=1).squeeze()

    return saliency_map


def compute_gradCAM_saliency(policy, img, skill_idx, layer=True):
    if layer:
        gc = LayerGradCam(policy, policy.prior_encoder.resnet.layer4[-1].conv2)
        attr = gc.attribute(img, skill_idx.item())
        attr = torch.nn.functional.interpolate(attr, size=img.shape[-2:], mode='bilinear', align_corners=True)
        attr = attr.squeeze().detach().cpu().numpy()
    else:
        gc = GuidedGradCam(policy, policy.prior_encoder.resnet.layer4[-1].conv2)
        attr = gc.attribute(img, skill_idx.item()).detach().cpu().numpy()
        attr = np.linalg.norm(attr, ord=2, axis=1).squeeze()

    return attr


def compute_deeplift_saliency(policy, img, skill_idx):
    dl = DeepLift(policy)
    attr = dl.attribute(img, target=skill_idx.item()).detach().cpu().numpy()
    saliency_map = np.linalg.norm(attr, ord=2, axis=1).squeeze()

    return saliency_map


def compute_deepSHAP_saliency(policy, img, skill_idx):
    dls = DeepLiftShap(policy)

    baseline1 = torch.zeros_like(img).squeeze()
    baseline2 = torch.ones_like(img).squeeze()
    baseline3 = img.mean(dim=0, keepdim=True).expand_as(img).squeeze()

    baselines = torch.stack([baseline1, baseline2, baseline3])
    attr = dls.attribute(img, baselines=baselines,
                         target=skill_idx.item()).detach().cpu().numpy()

    saliency_map = np.linalg.norm(attr, ord=2, axis=1).squeeze()
    return saliency_map


def compute_guided_backprop_saliency(policy, img, skill_idx):
    gbp = GuidedBackprop(policy)
    attr = gbp.attribute(img, target=skill_idx.item()).detach().cpu().numpy()
    saliency_map = np.linalg.norm(attr, ord=2, axis=1).squeeze().squeeze()
    return saliency_map


def compute_gradient_saliency(policy, img, skill_index):
    """
    计算输入图像对embedding各维度的梯度显著性
    返回：
        grads: (K, H, W) 的梯度张量，每个维度对应一个空间梯度图
    """
    img.requires_grad_(True)

    probs = policy(img)

    policy.zero_grad()
    grad_output = torch.zeros_like(probs)
    grad_output[0, skill_index] = 1.0  # 仅保留目标维度的梯度
    probs.backward(gradient=grad_output, retain_graph=True)
    # 提取输入图像的梯度
    grad = img.grad.data.cpu().numpy()
    saliency_map = np.linalg.norm(grad, ord=2, axis=1).squeeze()

    return saliency_map


def compute_integrated_gradient_saliency(policy, img, skill_index):
    """
    计算输入图像对embedding各维度的梯度显著性
    返回：
        grads: (K, H, W) 的梯度张量，每个维度对应一个空间梯度图
    """
    # img.requires_grad_(True)
    #
    # # 计算 Integrated Gradients
    # steps = 100
    #
    #
    # # K = probs.shape[1]
    # # saliency_maps = np.zeros((img.shape[2], img.shape[3]))  # (K, H, W)
    # baseline = torch.zeros_like(img).to(self.device)  # 选择全零图像作为 baseline
    # scaled_inputs = [(baseline + (float(i) / steps) * (img - baseline)) for i in range(steps)]
    #
    # grads = []
    # for img in scaled_inputs:
    #     img = img.detach().requires_grad_(True)
    #     probs = policy.compute_learned_prior(img).dist.probs
    #     target = probs[0, skill_index]  # 取第 j 维的概率值
    #
    #     policy.zero_grad()
    #     # grad_output = torch.zeros_like(probs)
    #     # grad_output[0, j] = 1.0
    #     target.backward()
    #     # grad = torch.autograd.grad(target, img, retain_graph=True)[0]  # 直接计算梯度
    #     # grads.append(grad.cpu().numpy())
    #
    #     grads.append(img.grad.data.detach().cpu().numpy())
    #
    # avg_grad = np.mean(grads, axis=0)  # 计算平均梯度
    # integrated_grads = (
    #                            img.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_grad  # 计算 IG
    #
    # # 计算 IG 显著性并存储
    # saliency_maps = np.linalg.norm(integrated_grads, ord=2, axis=1).squeeze()  # (H, W)
    #
    # return saliency_maps, probs.flatten().detach().cpu().numpy()  # (H, W)

    ig = IntegratedGradients(policy)
    saliency = ig.attribute(img, target=skill_index.item(),
                            return_convergence_delta=False).detach().cpu().numpy()
    saliency_maps = np.linalg.norm(saliency, ord=2, axis=1).squeeze()

    return saliency_maps


def compute_gradient_shap(policy, img, skill_index):
    gs = GradientShap(policy)
    baseline = torch.randn_like(img).to(device)  # 选择全零图像作为 baseline
    #
    attr = gs.attribute(img, target=skill_index.item(), baselines=baseline,
                        return_convergence_delta=False).detach().cpu().numpy()
    saliency_maps = np.linalg.norm(attr, ord=2, axis=1).squeeze()

    return saliency_maps


def compute_input_x_gradient_saliency(policy, img, skill_idx):
    input_x_gradient = InputXGradient(policy)
    attr = input_x_gradient.attribute(img, target=skill_idx.item()).squeeze().detach().cpu().numpy()
    saliency_maps = np.linalg.norm(attr, ord=2, axis=0).squeeze().squeeze()
    return saliency_maps


def compute_perturbation_saliency(policy, img, skill_idx, sigma=5, kernel_size=17, batch_size=1024):
    with torch.no_grad():
        policy.eval()
        H, W = img.shape[2], img.shape[3]

        probs_original_logits = policy(img)[:, skill_idx]  # (1, K)
        saliency_maps = torch.zeros((H, W), device=device)  # (K, H, W)

        all_pixels = [(i, j) for i in range(H) for j in range(W)]
        num_batches = (len(all_pixels) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_pixels = all_pixels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_size_actual = len(batch_pixels)

            perturbed_imgs = torch.zeros((batch_size_actual, *img.shape[1:]), device=device)

            for idx, (i, j) in enumerate(batch_pixels):
                perturbed_imgs[idx] = gaussian_blur_perturb(img, i, j, sigma, kernel_size)
                # perturbed_imgs[idx] = poisson_gaussian_noise_perturb(img, i, j, sigma)

            probs_perturbed_logits = policy(perturbed_imgs)[:, skill_idx]  # (B, K)

            # delta_probs = torch.norm(probs_original - probs_perturbed, p=2, dim=1)  # MSE LOSS
            # delta_probs = kl_categorical(probs_original_logits, probs_perturbed_logits)

            delta_probs = torch.where(
                probs_perturbed_logits > probs_original_logits,
                torch.exp(probs_perturbed_logits) * torch.expm1(probs_original_logits - probs_perturbed_logits),
                -torch.exp(probs_original_logits) * torch.expm1(probs_perturbed_logits - probs_original_logits)
            )

            # 存入 saliency map
            for idx, (i, j) in enumerate(batch_pixels):
                saliency_maps[i, j] = delta_probs[idx]

        return saliency_maps.cpu().numpy()


def compute_lime_saliency(policy, img, skill_idx, feature_mask=None):
    lime = Lime(policy)
    feature_mask = torch.from_numpy(feature_mask.astype(np.int32)).to(device=device).unsqueeze(0).repeat(1, 3, 1, 1)
    # print(torch.max(feature_mask))
    attr = lime.attribute(img, target=skill_idx.item(), feature_mask=feature_mask, n_samples=10000,
                          perturbations_per_eval=256,
                          show_progress=True).detach().cpu().numpy()
    saliency_maps = np.linalg.norm(attr, ord=2, axis=1).squeeze().squeeze()
    return saliency_maps


def compute_feature_ablation_saliency(policy, img, skill_idx, feature_mask=None):
    feature_ablation = FeatureAblation(policy)
    feature_mask = torch.from_numpy(feature_mask.astype(np.int32)).to(device=device).unsqueeze(0).repeat(1, 3, 1, 1)
    # print(torch.max(feature_mask))
    attr = feature_ablation.attribute(img, target=skill_idx.item(), feature_mask=feature_mask,
                                      show_progress=True).detach().cpu().numpy()
    saliency_maps = np.linalg.norm(attr, ord=2, axis=1).squeeze().squeeze()
    return saliency_maps


def compute_deconvolution_saliency(policy, img, skill_idx):
    deconvolution = Deconvolution(policy)
    attr = deconvolution.attribute(img, target=skill_idx.item()).squeeze().detach().cpu().numpy()
    saliency_maps = np.linalg.norm(attr, ord=2, axis=0).squeeze().squeeze()
    return saliency_maps


def compute_shapley_value_sampling_saliency(policy, img, skill_idx, feature_mask=None):
    svs = ShapleyValueSampling(policy)
    feature_mask = torch.from_numpy(feature_mask.astype(np.int32)).to(device=device).unsqueeze(0).repeat(1, 3, 1, 1)
    attr = svs.attribute(img, target=skill_idx.item(), feature_mask=feature_mask, n_samples=1000,
                         perturbations_per_eval=256,
                         show_progress=True).detach().cpu().numpy()
    saliency_maps = np.linalg.norm(attr, ord=2, axis=1).squeeze().squeeze()
    return saliency_maps


def compute_instance_influence(saliency, instance_masks, percentile=0.99):
    """
    计算每个实例对各个skill的贡献
    参数：
        grads: (H, W) 梯度显著性图
        instance_masks: List[(H, W) binary masks]
    返回：
        influence: (N_instances) 影响矩阵
    """
    influence = []
    baseline = saliency.mean()

    for name in instance_masks.keys():
        mask = instance_masks[name]
        # area = mask.sum() + 1e-6

        normal = (saliency - baseline) * mask[np.newaxis, :]  # (H, W)
        influence.append(normal.sum())
        # influence.append(saliency.sum() / area)

    return np.array(influence)

    # influence = []
    # for mask in instance_masks:
    #     mask = mask['mask'].squeeze(0)
    #     masked_grads = grads * mask[np.newaxis, :, :]  # (K, H, W)
    #
    #     valid_values = masked_grads[:, mask > 0]  # 仅选择 mask 位置的梯度值
    #     influence = np.percentile(valid_values, percentile, axis=1)
    #
    #     influence.append(influence)
    #
    # return np.array(influence)
