import copy
import datetime
import pickle

import cv2
import h5py
import numpy as np
import torch
import os
import imp

from captum.attr import IntegratedGradients, GradientShap, GuidedGradCam, LayerGradCam, DeepLift, DeepLiftShap, LRP, \
    Occlusion
from captum.attr._utils.lrp_rules import EpsilonRule
from matplotlib import pyplot as plt

from src.rl.components.params import get_args
from src.rl.utils.rollout import HPRolloutSaver
from src.train import set_seeds, make_path
from src.components.checkpointer import get_config_path
from src.utils.dist import kl_categorical
from src.utils.general import AttrDict
from instance_seg_test import generate_masks_with_sam, initialize_clip, classify_with_clip
from openai import OpenAI

from src.utils.image import gaussian_blur_perturb, poisson_gaussian_noise_perturb
from src.utils.llm import chain, prompt_template
from src.utils.pytorch import no_batchnorm_update
from src.utils.video import create_video_from_pdfs_and_markdowns
from src.utils.render import render_mujoco_object_masks
from src.utils.general import TopKMetricAverageMeter


class InstanceInfluence:
    """Sets up RL training loop, instantiates all components, runs training."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conf = conf = self.get_config()
        self.conf.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self._hp = self.conf.general  # override defaults with config file

        self.env = self._hp.environment(self.conf.env)
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, None, self._hp.max_rollout_len)

        self.labels = self.conf.data.dataset_spec.labels
        self.boxes = self.conf.data.dataset_spec.boxes
        self.gt_skill_index = self.conf.data.dataset_spec.gt_skill_index

        client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

        self.metric = TopKMetricAverageMeter()

        for i in range(self.args.n_episode):
            self.args.episode_idx = i
            self.init_dir()

            if self.metric.load_from_cache(self.args.save_dir) and not self.args.overwrite_cache:
                continue
            else:
                episode = self.sample()
                # create_video_from_pdfs_and_markdowns(num_files=25, pdf_dir=self.args.save_dir, md_dir=self.args.save_dir,
                #                                      output_dir=self.args.save_dir, output_video='explanation.mp4',
                #                                      language_output=False, clean_tmp=True)
                self.analyze(episode)

        print(self.metric.compute())

    def init_dir(self):
        self.args.save_dir = os.path.join(self.conf.exp_path, 'explanation', self.args.exp_method, f'episode_{self.args.episode_idx}')
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        print(f'save_dir: {self.args.save_dir}')

    def sample(self, target_subtask=4):
        sample_rollout_path = os.path.join(self.conf.exp_path, f"sample_rollout_{self.args.episode_idx}.h5")
        if os.path.exists(sample_rollout_path):
            with h5py.File(sample_rollout_path, 'r') as f:
                print(f"Found existing sample rollout: {sample_rollout_path}")
                episode_data = {}
                if 'states' in f:
                    episode_data['observation'] = f['states'][:]
                if 'actions' in f:
                    episode_data['action'] = f['actions'][:]
                if 'is_hl_step' in f:
                    episode_data['is_hl_step'] = f['is_hl_step'][:]
                if 'hl_action_index' in f:
                    episode_data['hl_action_index'] = f['hl_action_index'][:]

                print(f"Episode data keys: {list(episode_data.keys())}")
                return episode_data

        saver = HPRolloutSaver(self.conf.exp_path, hl_only=False)
        reward = 0

        while reward < target_subtask:
            with self.agent.val_mode():
                with torch.no_grad():
                    episode = self.sampler.sample_episode(is_train=False, render=False)
                    reward = np.array(episode.reward).sum()

                print(reward)

        saver.save_rollout(episode)
        saver.save(f"sample_rollout_{self.args.episode_idx}", save_interval=1, reset=True)
        return episode

    def analyze(self, episode):
        """Generate rollouts and save to hdf5 files."""

        # initialize clip model
        # clip_model, clip_processor = initialize_clip(device=self.device)

        # encoder = self.agent.hl_agent.policy.net.img_encoder_p
        policy = self.agent.hl_agent.policy
        # num_skill = self.conf.agent.hl_agent_params.policy_params.skill_dim
        hl_step = 0
        result = []

        for i in range(len(episode['observation'])):
            if episode['is_hl_step'][i]:
                print(f"hl step {hl_step}")
                skill_index = episode['hl_action_index'][i]
                # current_task = episode['info'][i][0]['current_task']

                obs = episode['observation'][i]
                with torch.no_grad():
                    obs = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                    obs = self.agent.hl_agent.policy.net.unflatten_obs(obs)
                    image_obs = obs.prior_obs
                    state = obs.obs

                    print(f"Skill Index: {skill_index}")

                with no_batchnorm_update(policy):
                    processed_img = image_obs.float()

                    img = (processed_img * 0.5 + 0.5) * 255.0
                    img = img.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.uint8)
                    # self.save_img(img, index)

                    if self.args.real_segmentation:
                        masks = render_mujoco_object_masks(self.env, state.squeeze().cpu().numpy())
                    else:
                        # generate masks with names
                        masks = generate_masks_with_sam(img, self.boxes, self.args.save_dir)

                    # results = classify_with_clip(clip_model, clip_processor, img, masks, self.candidate_labels)

                    saliency, dist = self.compute_saliency(policy.net, processed_img, skill_index)
                    saliency = self._normalize_saliency(saliency)
                    influence = self.compute_instance_influence(saliency, masks)  # (N, K)

                self.visualize_influence(influence, dist, hl_step, skill_index)
                self.visualize_dimension_influence(img, masks, saliency, influence, dist, skill_index, hl_step)

                influence_str = ""
                for i, (label, obj_influence) in enumerate(
                        zip(self.labels, influence)):
                    influence_str += f'Object {i} ({label}): {obj_influence:.3f}\n'

                # print(influence_str)
                eval = self.evaluate(influence, skill_index, self.gt_skill_index)
                result.append(eval)
                self.metric.update(eval)
                # print(result)

                # explanation = chain.invoke(
                #     prompt_template.format(skill_index=skill_index, current_task=current_task, score=influence_str))
                # print(explanation)
                # self.save_explanation(explanation, hl_step)

                hl_step += 1

        self.save_result_to_cache(result)
        create_video_from_pdfs_and_markdowns(num_files=hl_step, pdf_dir=self.args.save_dir, md_dir=self.args.save_dir,
                                             output_dir=self.args.save_dir, output_video='explanation.mp4',
                                             language_output=False, clean_tmp=True)

    def save_img(self, img, index):
        # image = cv2.imread(img)
        plt.imshow(img)
        plt.savefig(f'{self.args.save_dir}/original_img_{index}.png')
        plt.close()

    def save_explanation(self, explanation, index):
        file_path = f'{self.args.save_dir}/explanation_{index}.md'

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(explanation)

    def save_result_to_cache(self, result):
        with open(os.path.join(self.args.save_dir, 'result.pkl'), "wb") as f:
            pickle.dump(result, f)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration

        conf.agent = conf_module.agent_config
        conf.agent.device = self.device

        # data config
        conf.data = conf_module.data_config

        # environment config
        conf.env = conf_module.env_config
        conf.env.device = self.device  # add device to env config as it directly returns tensors

        # sampler config
        conf.sampler = conf_module.sampler_config if hasattr(conf_module, 'sampler_config') else AttrDict({})

        # model loading config
        conf.ckpt_path = conf.agent.checkpt_path if 'checkpt_path' in conf.agent else None

        return conf

    def visualize_influence(self, influence, dist, index, skill_index):
        plt.figure(figsize=(7, 3))
        # K = dist.shape[0]

        plt.bar(self.labels, influence)
        # plt.colorbar()
        plt.title(f'Instance Influence (Skill {skill_index}, prob={dist[skill_index].item():.2f})')
        plt.xlabel('Attr')
        plt.ylabel('Objects')
        # plt.yticks(ticks=np.arange(len(self.candidate_labels)), labels=self.candidate_labels)
        # for i in range(K):
        #     plt.text(i, 0, f"{dist[i]:.2f}", ha="center", va="center", color="w", size=7)
        plt.tight_layout()
        plt.savefig(f'{self.args.save_dir}/instance_influence_{index}.pdf')
        plt.close()

    def compute_saliency(self, policy, processed_img, skill_idx):
        if self.args.exp_method == 'ig':
            return self.compute_integrated_gradient_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'gradient':
            return self.compute_gradient_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'gradient_shap':
            return self.compute_gradient_shap(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'gaussian_perturbation':
            return self.compute_perturbation_saliency(policy, processed_img)
        elif self.args.exp_method == 'grad_cam':
            return self.compute_gradCAM_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'guided_grad_cam':
            return self.compute_gradCAM_saliency(policy, processed_img, skill_idx, layer=False)
        elif self.args.exp_method == 'deep_lift':
            return self.compute_deeplift_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'deep_shap':
            return self.compute_deepSHAP_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'lrp':
            return self.compute_lrp_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'occlusion':
            return self.compute_occlusion_saliency(policy, processed_img, skill_idx)
        else:
            raise Exception(f'unsupported explanation method {self.args.exp_method}')

    def compute_lrp_saliency(self, policy, processed_img, skill_idx):
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

        policy = copy.deepcopy(policy)
        remove_logsoftmax(policy.prior_head)
        lrp = LRP(policy)
        attr = lrp.attribute(processed_img, target=skill_idx.item()).squeeze().detach().cpu().numpy().mean(axis=0)

        probs = policy.compute_learned_prior(processed_img).dist.probs
        return attr, probs.flatten().detach().cpu().numpy()

    def compute_occlusion_saliency(self, policy, processed_img, skill_idx):
        occ = Occlusion(policy)
        attr = occ.attribute(processed_img, target=skill_idx.item(),
                             sliding_window_shapes=(1, 5, 5)).squeeze().detach().cpu().numpy().mean(axis=0)

        probs = policy.compute_learned_prior(processed_img).dist.probs
        return attr, probs.flatten().detach().cpu().numpy()

    def compute_gradCAM_saliency(self, policy, processed_img, skill_idx, layer=True):
        if layer:
            gc = LayerGradCam(policy, policy.prior_encoder.resnet.layer4[-1].conv2)
            attr = gc.attribute(processed_img, skill_idx.item())
            attr = torch.nn.functional.interpolate(attr, size=processed_img.shape[-2:], mode='bilinear',
                                                   align_corners=True)
            attr = attr.squeeze().detach().cpu().numpy()
        else:
            gc = GuidedGradCam(policy, policy.prior_encoder.resnet.layer4[-1].conv2)
            attr = gc.attribute(processed_img, skill_idx.item()).squeeze().detach().cpu().numpy().mean(axis=0)

        probs = policy.compute_learned_prior(processed_img).dist.probs
        return attr, probs.flatten().detach().cpu().numpy()

    def compute_deeplift_saliency(self, policy, processed_img, skill_idx):
        dl = DeepLift(policy)
        attr = dl.attribute(processed_img, target=skill_idx.item()).squeeze().detach().cpu().numpy().mean(axis=0)
        probs = policy.compute_learned_prior(processed_img).dist.probs

        return attr, probs.flatten().detach().cpu().numpy()

    def compute_deepSHAP_saliency(self, policy, processed_img, skill_idx):
        dls = DeepLiftShap(policy)

        baseline1 = torch.zeros_like(processed_img).squeeze()
        baseline2 = torch.ones_like(processed_img).squeeze()
        baseline3 = processed_img.mean(dim=0, keepdim=True).expand_as(processed_img).squeeze()

        baselines = torch.stack([baseline1, baseline2, baseline3])
        attr = dls.attribute(processed_img, baselines=baselines,
                             target=skill_idx.item()).squeeze().detach().cpu().numpy().mean(axis=0)
        probs = policy.compute_learned_prior(processed_img).dist.probs

        return attr, probs.flatten().detach().cpu().numpy()

    def compute_gradient_saliency(self, policy, processed_img, skill_index):
        """
        计算输入图像对embedding各维度的梯度显著性
        返回：
            grads: (K, H, W) 的梯度张量，每个维度对应一个空间梯度图
        """
        processed_img.requires_grad_(True)

        probs = policy(processed_img)

        policy.zero_grad()
        grad_output = torch.zeros_like(probs)
        grad_output[0, skill_index] = 1.0  # 仅保留目标维度的梯度
        probs.backward(gradient=grad_output, retain_graph=True)
        # 提取输入图像的梯度
        grad = processed_img.grad.data.cpu().numpy()
        saliency_map = np.linalg.norm(grad, ord=2, axis=1).squeeze()

        return saliency_map, probs.flatten().detach().cpu().numpy()  # (H, W)

    def compute_integrated_gradient_saliency(self, policy, processed_img, skill_index):
        """
        计算输入图像对embedding各维度的梯度显著性
        返回：
            grads: (K, H, W) 的梯度张量，每个维度对应一个空间梯度图
        """
        # processed_img.requires_grad_(True)
        #
        # # 计算 Integrated Gradients
        # steps = 100
        #
        probs = policy.compute_learned_prior(processed_img).dist.probs
        #
        # # K = probs.shape[1]
        # # saliency_maps = np.zeros((processed_img.shape[2], processed_img.shape[3]))  # (K, H, W)
        # baseline = torch.zeros_like(processed_img).to(self.device)  # 选择全零图像作为 baseline
        # scaled_inputs = [(baseline + (float(i) / steps) * (processed_img - baseline)) for i in range(steps)]
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
        #                            processed_img.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_grad  # 计算 IG
        #
        # # 计算 IG 显著性并存储
        # saliency_maps = np.linalg.norm(integrated_grads, ord=2, axis=1).squeeze()  # (H, W)
        #
        # return saliency_maps, probs.flatten().detach().cpu().numpy()  # (H, W)

        ig = IntegratedGradients(policy)
        # gs = GradientShap(policy)
        # baseline = torch.zeros_like(processed_img).to(self.device)  # 选择全零图像作为 baseline
        #
        saliency = ig.attribute(processed_img, target=skill_index.item(),
                                        return_convergence_delta=False).detach().cpu().numpy()
        # saliency = gs.attribute(processed_img, baseline, target=j,
        #                         return_convergence_delta=False).detach().cpu().numpy()
        saliency_maps = np.linalg.norm(saliency, ord=2, axis=1).squeeze().squeeze()

        return saliency_maps, probs.flatten().detach().cpu().numpy()  # (H, W)

    def compute_gradient_shap(self, policy, processed_img, skill_index):
        probs = policy.compute_learned_prior(processed_img).dist.probs

        gs = GradientShap(policy)
        baseline = torch.randn_like(processed_img).to(self.device)  # 选择全零图像作为 baseline
        #
        saliency = gs.attribute(processed_img, target=skill_index.item(), baselines=baseline,
                                return_convergence_delta=False).detach().cpu().numpy()
        saliency_maps = np.linalg.norm(saliency, ord=2, axis=1).squeeze().squeeze()

        return saliency_maps, probs.flatten().detach().cpu().numpy()  # (H, W)

    def compute_perturbation_saliency(self, policy, processed_img, sigma=3, kernel_size=11, batch_size=1024):
        with torch.no_grad():
            # processed_img = processed_img.clone().to(self.device)
            H, W = processed_img.shape[2], processed_img.shape[3]

            probs_original_logits = policy.compute_learned_prior(processed_img).dist.logits  # (1, K)
            K = probs_original_logits.shape[1]

            saliency_maps = torch.zeros((H, W), device=self.device)  # (K, H, W)

            all_pixels = [(i, j) for i in range(H) for j in range(W)]
            num_batches = (len(all_pixels) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                batch_pixels = all_pixels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch_size_actual = len(batch_pixels)

                perturbed_imgs = torch.zeros((batch_size_actual, *processed_img.shape[1:]), device=self.device)

                # 批量计算扰动
                for idx, (i, j) in enumerate(batch_pixels):
                    perturbed_imgs[idx] = gaussian_blur_perturb(processed_img, i, j, sigma, kernel_size)
                    # perturbed_imgs[idx] = poisson_gaussian_noise_perturb(processed_img, i, j, sigma)

                probs_perturbed_logits = policy.compute_learned_prior(perturbed_imgs).dist.logits  # (B, K)

                # delta_probs = torch.norm(probs_original - probs_perturbed, p=2, dim=1)  # MSE LOSS
                delta_probs = kl_categorical(probs_original_logits, probs_perturbed_logits)

                # 存入 saliency map
                for idx, (i, j) in enumerate(batch_pixels):
                    saliency_maps[i, j] = delta_probs[idx]

                torch.cuda.empty_cache()

            return saliency_maps.cpu().numpy(), torch.exp(probs_original_logits).flatten().detach().cpu().numpy()

    def compute_instance_influence(self, saliency, instance_masks, percentile=0.99):

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

    def visualize_dimension_influence(self, img, masks, saliency, influence, dist, skill_index, hl_step):
        """
        可视化指定维度受实例影响的情况
        """
        topk = 5
        # num_instances = len(masks)
        # instance_names = list(masks.keys())

        fig = plt.figure(figsize=(16, 8))

        # === 第1列: 原图 ===
        plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        # === 第2列: Saliency Map ===
        plt.subplot(2, 4, 2)
        plt.imshow(saliency, cmap='jet')
        plt.colorbar()
        plt.title(f"Gradient Map (Skill: {self.labels[skill_index]}, p={dist[skill_index].item():.2f})")
        plt.axis('off')

        # === 第3列: Overlay ===
        plt.subplot(2, 4, 3)
        grad_colored = plt.get_cmap('jet')(saliency)
        plt.imshow(img)
        plt.imshow(grad_colored, alpha=0.7)
        plt.title(f"Overlay")
        plt.axis('off')

        # === 第4列: Instance 编号图 ===
        plt.subplot(2, 4, 4)
        plt.imshow(img)
        for j, name in enumerate(masks.keys()):
            mask = masks[name]
            mask_colored = img.copy()
            mask_colored[mask > 0] = [255, 0, 0]
            plt.imshow(mask_colored, alpha=0.4)

            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                plt.text(cx, cy,
                         f"{j + 1}",
                         color='yellow', fontsize=10, weight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle="circle,pad=0.3", fc="black", ec="yellow", lw=1.5))
        plt.title("Instance Masks (Numbered)")
        plt.axis('off')

        # === 第5–6列: 编号 → 名称对照 ===
        plt.subplot(2, 4, (5, 6))
        plt.axis('off')

        y = 1.0
        dy = 1.0 / (len(self.labels) + 1)

        for j, name in enumerate(self.labels):
            text = f"{j + 1} {name}"

            if skill_index is not None and j == skill_index:
                plt.text(0, y, text,
                         fontsize=15, weight='bold', color='gold',
                         backgroundcolor='black', va='top', family='monospace')
            else:
                plt.text(0, y, text,
                         fontsize=14, color='black',
                         va='top', family='monospace')

            y -= dy

        plt.title("Objects", fontsize=14)

        # === 第7–8列: Top-K Influence 柱状图 ===
        plt.subplot(2, 4, (7, 8))
        influence_arr = np.array(influence)
        top_indices = np.argsort(influence_arr)[::-1][:topk]
        top_values = influence_arr[top_indices]
        top_names = [f"{self.labels[i]}" for i in top_indices]

        bars = plt.bar(top_names, top_values, color=plt.cm.viridis(np.linspace(0.3, 0.9, topk)))
        plt.title(f"Top {topk} Influential Instances", fontsize=14)
        plt.xlabel("Object Index")
        plt.ylabel("Influence Score")

        for rect, val in zip(bars, top_values):
            plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                     f"{val:.2f}", ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.args.save_dir}/skill_influence_{hl_step}.pdf')
        plt.close()

    def _normalize_saliency(self, saliency):
        saliency_min = saliency.min()
        saliency_max = saliency.max()
        normalized_saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return normalized_saliency

    def evaluate(self, influence, ground_truth, task_gt_objects, topk=4):
        """
        评估 saliency 分布在 top-1 / top-k 情况下对 ground truth 的命中情况

        参数:
            influence: List[float]，每个物体的 saliency 值（顺序与 self.labels 对应）
            ground_truth: str，当前应操作的目标物体
            task_gt_objects: List[str]，整个任务应涉及的 4 个关键物体名称
            topk: int，考虑 top-k 的范围

        返回:
            dict，包括 top-1 准确率、top-k recall、strict 命中等
        """
        assert len(influence) == len(self.labels), "影响值和标签数量不一致"

        # 获取排序后的索引和对应物体
        indices = sorted(range(len(influence)), key=lambda i: -influence[i])
        # ranked_objects = [self.labels[i] for i in indices]
        # topk_objects = ranked_objects[:topk]

        # Top-1 准确性
        top1_hit = int(indices[0] == ground_truth)

        # Top-k Recall：命中几个任务目标物体
        correct_topk = len(set(indices[:topk]) & set(task_gt_objects))
        topk_recall = correct_topk / len(task_gt_objects)

        # Strict Top-k 是否完全覆盖所有任务物体
        # strict_topk = int(set(task_gt_objects).issubset(set(indices[:topk])))

        return {
            "top1": top1_hit,
            f"top{topk}_recall": topk_recall,
            # f"strict_top{topk}": strict_topk,
        }

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_exp_dir(self):
        return os.environ['EXP_DIR']


if __name__ == '__main__':
    InstanceInfluence(args=get_args())
