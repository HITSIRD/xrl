import copy

import numpy as np
import torch
from captum.attr import IntegratedGradients, GradientShap, GuidedGradCam, LayerGradCam, DeepLift, DeepLiftShap, LRP, \
    Occlusion, GuidedBackprop, InputXGradient, Lime, FeaturePermutation, FeatureAblation, Deconvolution, \
    ShapleyValueSampling

from src.utils.dist import kl_categorical
from src.utils.image import gaussian_blur_perturb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_hidden_saliency(exp_method, policy, hidden_state, processed_img, skill_idx):
    with policy.enable_hidden_analysis():
        policy.set_image(processed_img)
        if exp_method == 'ig':
            return compute_integrated_gradient_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'gradient':
            return compute_gradient_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'gradient_shap':
            return compute_gradient_shap(policy, hidden_state, skill_idx)
        elif exp_method == 'gaussian_perturbation':
            return compute_perturbation_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'guided_backprop':
            return compute_guided_backprop_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'input_x_gradient':
            return compute_input_x_gradient_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'grad_cam':
            return compute_gradCAM_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'guided_grad_cam':
            return compute_gradCAM_saliency(policy, hidden_state, skill_idx, layer=False)
        elif exp_method == 'deep_lift':
            return compute_deeplift_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'deep_shap':
            return compute_deepSHAP_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'lrp':
            return compute_lrp_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'occlusion':
            return compute_occlusion_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'lime':
            return compute_lime_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'feature_ablation':
            return compute_feature_ablation_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'deconvolution':
            return compute_deconvolution_saliency(policy, hidden_state, skill_idx)
        elif exp_method == 'svs':
            return compute_shapley_value_sampling_saliency(policy, hidden_state, skill_idx)
        else:
            raise Exception(f'unsupported explanation method {exp_method}')


def compute_lrp_saliency(policy, hidden_state, skill_idx):
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
    attr = lrp.attribute(hidden_state, target=skill_idx.item()).detach().cpu().numpy()

    return attr


def compute_occlusion_saliency(policy, hidden_state, skill_idx):
    occ = Occlusion(policy)
    attr = occ.attribute(hidden_state, target=skill_idx.item(), sliding_window_shapes=(5,), perturbations_per_eval=256,
                         show_progress=True).detach().cpu().numpy()

    return attr


def compute_gradCAM_saliency(policy, hidden_state, skill_idx, layer=True):
    if layer:
        gc = LayerGradCam(policy, policy.prior_encoder.resnet.layer4[-1].conv2)
        attr = gc.attribute(hidden_state, skill_idx.item())
        attr = torch.nn.functional.interpolate(attr, size=hidden_state.shape[-2:], mode='bilinear', align_corners=True)
    else:
        gc = GuidedGradCam(policy, policy.prior_encoder.resnet.layer4[-1].conv2)
        attr = gc.attribute(hidden_state, skill_idx.item()).detach().cpu().numpy()
    return attr


def compute_deeplift_saliency(policy, hidden_state, skill_idx):
    dl = DeepLift(policy)
    attr = dl.attribute(hidden_state, target=skill_idx.item()).detach().cpu().numpy()
    return attr


def compute_deepSHAP_saliency(policy, hidden_state, skill_idx):
    dls = DeepLiftShap(policy)

    baseline1 = torch.zeros_like(hidden_state).squeeze()
    baseline2 = torch.ones_like(hidden_state).squeeze()
    baseline3 = hidden_state.mean(dim=0, keepdim=True).expand_as(hidden_state).squeeze()

    baselines = torch.stack([baseline1, baseline2, baseline3])
    attr = dls.attribute(hidden_state, baselines=baselines,
                         target=skill_idx.item()).detach().cpu().numpy()

    return attr


def compute_guided_backprop_saliency(policy, hidden_state, skill_idx):
    gbp = GuidedBackprop(policy)
    attr = gbp.attribute(hidden_state, target=skill_idx.item()).detach().cpu().numpy()
    return attr


def compute_gradient_saliency(policy, hidden_state, skill_index):
    probs = policy(hidden_state)

    policy.zero_grad()
    grad_output = torch.zeros_like(probs)
    grad_output[0, skill_index] = 1.0  # 仅保留目标维度的梯度
    probs.backward(gradient=grad_output, retain_graph=True)
    # 提取输入图像的梯度
    grad = hidden_state.grad.data.cpu().numpy()

    return grad


def compute_integrated_gradient_saliency(policy, hidden_state, skill_index):
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
    saliency = ig.attribute(hidden_state, target=skill_index.item(),
                            return_convergence_delta=False).detach().cpu().numpy()
    return saliency


def compute_gradient_shap(policy, hidden_state, skill_index):
    gs = GradientShap(policy)
    baseline = torch.randn_like(hidden_state).to(device)  # 选择全零图像作为 baseline
    #
    attr = gs.attribute(hidden_state, target=skill_index.item(), baselines=baseline,
                        return_convergence_delta=False).detach().cpu().numpy()

    return attr


def compute_input_x_gradient_saliency(policy, hidden_state, skill_idx):
    input_x_gradient = InputXGradient(policy)
    attr = input_x_gradient.attribute(hidden_state, target=skill_idx.item()).squeeze().detach().cpu().numpy()
    return attr


def compute_perturbation_saliency(policy, hidden_state, skill_idx, sigma=5, kernel_size=17, batch_size=1024):
    with torch.no_grad():
        # img = img.clone().to(self.device)
        H = hidden_state.shape[1]

        probs_original_logits = policy.compute_learned_prior(hidden_state).dist.logits[:, skill_idx]  # (1, K)
        saliency_maps = torch.zeros((H, W), device=device)  # (K, H, W)

        all_pixels = [(i) for i in range(H)]
        num_batches = (len(all_pixels) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_pixels = all_pixels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_size_actual = len(batch_pixels)

            perturbed_imgs = torch.zeros((batch_size_actual, *hidden_state.shape[1:]), device=device)

            # 批量计算扰动
            for idx, (i) in enumerate(batch_pixels):
                perturbed_imgs[idx] = gaussian_blur_perturb(hidden_state, i, j, sigma, kernel_size)
                # perturbed_imgs[idx] = poisson_gaussian_noise_perturb(img, i, j, sigma)

            probs_perturbed_logits = policy.compute_learned_prior(perturbed_imgs).dist.logits[:, skill_idx]  # (B, K)

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


def compute_lime_saliency(policy, hidden_state, skill_idx, feature_mask=None):
    lime = Lime(policy)
    attr = lime.attribute(hidden_state, target=skill_idx.item(), n_samples=1000, perturbations_per_eval=256,
                          show_progress=True).detach().cpu().numpy()
    return attr


def compute_feature_ablation_saliency(policy, hidden_state, skill_idx, feature_mask=None):
    feature_ablation = FeatureAblation(policy)
    attr = feature_ablation.attribute(hidden_state, target=skill_idx.item()).detach().cpu().numpy()
    return attr


def compute_deconvolution_saliency(policy, hidden_state, skill_idx):
    deconvolution = Deconvolution(policy)
    attr = deconvolution.attribute(hidden_state, target=skill_idx.item()).squeeze().detach().cpu().numpy()
    return attr


def compute_shapley_value_sampling_saliency(policy, hidden_state, skill_idx, feature_mask=None):
    svs = ShapleyValueSampling(policy)
    attr = svs.attribute(hidden_state, target=skill_idx.item(), n_samples=1000, perturbations_per_eval=256,
                         show_progress=True).detach().cpu().numpy()
    return attr
