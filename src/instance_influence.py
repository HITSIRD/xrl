import datetime

import cv2
import h5py
import numpy as np
import torch
import os
import imp
import json

from captum.attr import IntegratedGradients, GradientShap
from flax.core.nn import embedding
from matplotlib import pyplot as plt

from src.rl.components.params import get_args
from src.train import set_seeds, make_path
from src.components.checkpointer import get_config_path
from src.utils.dist import kl_categorical
from src.utils.general import AttrDict, ParamDict, AverageTimer, timing, pretty_print
# from src.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks, mpi_sum, mpi_gather_experience
from src.rl.components.sampler import Sampler
from src.rl.components.buffer import RolloutStorage
from instance_seg_test import generate_masks_with_sam, initialize_clip, classify_with_clip
from openai import OpenAI

from src.utils.image import gaussian_blur_perturb, poisson_gaussian_noise_perturb
from src.utils.llm import chain, prompt_template
from src.utils.pytorch import no_batchnorm_update
from src.utils.video import create_video_from_pdfs_and_markdowns


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

        self.candidate_labels = ["microwave",
                                 "slide cabinet",
                                 "hinge cabinet",
                                 "light switch",
                                 "top burner switch",
                                 "bottom burner switch",
                                 "kettle"]

        client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

        self.analyze()

    def _default_hparams(self):
        default_dict = ParamDict({
            'seed': None,
            'agent': None,
            'data_dir': None,  # directory where dataset is in
            'sampler': Sampler,  # sampler type used
            'exp_path': None,  # Path to the folder with experiments
            'dataset_path': 'experiments/hrl/kitchen/cdt_cl_vq_prior_cdt/mkbl_d6_s1_avgprob/fine_500_50.h5',
        })
        return default_dict

    def analyze(self):
        """Generate rollouts and save to hdf5 files."""
        # sample an episode
        reward = 0

        val_rollout_storage = RolloutStorage()

        while reward < 4:
            with self.agent.val_mode():
                with torch.no_grad():
                    # deterministic policy
                    # episode = self.sampler.sample_episode(index=i, is_train=False, render=False, task=False)

                    # spirl_cl_vq & tree policy
                    episode = self.sampler.sample_episode(is_train=False, render=False)

                    # val_rollout_storage.append(episode, reward_only=True)
                    val_rollout_storage.append(episode)
                    reward = np.array(episode.reward).sum()

                # episode_reward_mean, episode_reward_std = val_rollout_storage.rollout_stats(std=True)
                # complete_task, count = val_rollout_storage.evaluate_task()

                print(reward)

        # initialize clip model
        clip_model, clip_processor = initialize_clip(device=self.device)

        # encoder = self.agent.hl_agent.policy.net.img_encoder_p
        policy = self.agent.hl_agent.policy
        num_skill = self.conf.agent.hl_agent_params.policy_params.skill_dim
        hl_step = 0

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

                    # output = policy(obs)
                    # prediction, dist, prob = output['action_index'], output['dist'], output['probs']
                    # print(f"Prediction: Skill {prediction}")
                    # print(f"Dist: {dist}")
                    print(f"Skill Index: {skill_index}")

                influence_str = ""

                with no_batchnorm_update(policy):
                    processed_img = image_obs.float()

                    # processed_img = image_obs.permute(0, 3, 1, 2).float()
                    # processed_img = processed_img / 255 * 2 - 1
                    # image_embedding = encoder(processed_img)

                    img = (processed_img * 0.5 + 0.5) * 255.0
                    img = img.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.uint8)
                    # self.save_img(img, index)

                    masks = generate_masks_with_sam(img, state)
                    # results = classify_with_clip(clip_model, clip_processor, img, masks, self.candidate_labels)

                    # grads, dist = self.compute_perturbation_saliency(policy.net, processed_img)  # (K, H, W)
                    grads, dist = self.compute_gradient_saliency(policy.net, processed_img)  # (K, H, W)
                    influence_matrix = self.compute_instance_influence(grads, masks)  # (N, K)
                    self.visualize_influence(influence_matrix, dist, hl_step)

                    # print("influence matrix shape: ", influence_matrix.shape)
                    # print(influence_matrix)

                    self.visualize_dimension_influence(img, masks, grads, influence_matrix, dist, num_skill,
                                                       skill_index, hl_step)

                    for i, (label, influence) in enumerate(
                            zip(self.candidate_labels, influence_matrix[:, skill_index])):
                        influence_str += f'Object {i} ({label}): {influence * 1000:.3f}\n'

                    print(influence_str)

                    # explanation = chain.invoke(
                    #     prompt_template.format(skill_index=skill_index, current_task=current_task, score=influence_str))
                    # print(explanation)
                    # self.save_explanation(explanation, hl_step)

                hl_step += 1

        create_video_from_pdfs_and_markdowns(num_files=hl_step, pdf_dir=self.conf.exp_path, md_dir=self.conf.exp_path,
                                             output_dir=self.conf.exp_path, output_video='explanation.mp4')

    def save_img(self, img, index):
        # image = cv2.imread(img)
        plt.imshow(img)
        plt.savefig(f'{self.args.exp_dir}/original_img_{index}.png')

    def save_explanation(self, explanation, index):
        file_path = f'{self.conf.exp_path}/explanation_{index}.md'

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(explanation)

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

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def visualize_influence(self, matrix, dist, index):
        plt.figure(figsize=(7, 3))
        K = dist.shape[0]

        plt.imshow(matrix * dist, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Instance Influence')
        plt.xlabel('Skills')
        plt.ylabel('Objects')
        plt.yticks(ticks=np.arange(len(self.candidate_labels)), labels=self.candidate_labels)
        for i in range(K):
            plt.text(i, 0, f"{dist[i]:.2f}", ha="center", va="center", color="w", size=7)
        plt.tight_layout()
        plt.savefig(f'{self.conf.exp_path}/instance_influence_{index}.pdf')

    def compute_gradient_saliency(self, policy, processed_img):
        """
        计算输入图像对embedding各维度的梯度显著性
        返回：
            grads: (K, H, W) 的梯度张量，每个维度对应一个空间梯度图
        """
        processed_img.requires_grad_(True)

        # probs = policy(processed_img)
        #
        # grads = []
        # for j in range(probs.shape[1]):
        #     policy.zero_grad()
        #     # 反向传播计算单个维度的梯度
        #     grad_output = torch.zeros_like(probs)
        #     grad_output[0, j] = 1.0  # 仅保留目标维度的梯度
        #     probs.backward(gradient=grad_output, retain_graph=True)
        #     # 提取输入图像的梯度
        #     grad = processed_img.grad.data.abs().sum(dim=1).squeeze()  # 合并通道维度
        #     grads.append(grad.cpu().numpy())
        #
        # return np.stack(grads, axis=0), probs.flatten()  # (K, H, W)

        # 计算 Integrated Gradients
        steps = 100

        # 获取 skill 维度 K
        probs = policy.compute_learned_prior(processed_img).dist.probs
        K = probs.shape[1]

        # # 存储所有 skill 的 IG 显著性图
        saliency_maps = np.zeros((K, processed_img.shape[2], processed_img.shape[3]))  # (K, H, W)
        baseline = torch.zeros_like(processed_img).to(self.device)  # 选择全零图像作为 baseline
        scaled_inputs = [(baseline + (float(i) / steps) * (processed_img - baseline)) for i in range(steps)]

        # 对每个 skill 计算 IG
        for j in range(K):
            grads = []
            for img in scaled_inputs:
                img = img.detach().requires_grad_(True)
                probs = policy.compute_learned_prior(img).dist.probs
                target = probs[0, j]  # 取第 j 维的概率值

                policy.zero_grad()
                # grad_output = torch.zeros_like(probs)
                # grad_output[0, j] = 1.0
                target.backward()
                # grad = torch.autograd.grad(target, img, retain_graph=True)[0]  # 直接计算梯度
                # grads.append(grad.cpu().numpy())

                grads.append(img.grad.data.detach().cpu().numpy())

            avg_grad = np.mean(grads, axis=0)  # 计算平均梯度
            integrated_grads = (processed_img.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_grad  # 计算 IG

            # 计算 IG 显著性并存储
            # saliency_maps[j] = linalg.norm(integrated_grads).mean(axis=1).squeeze()  # (H, W)
            saliency_maps[j] = np.linalg.norm(integrated_grads, ord=2, axis=1).squeeze()  # (H, W)

        return saliency_maps, probs.flatten().detach().cpu().numpy()  # (K, H, W)

        # ig = IntegratedGradients(policy)
        # # gs = GradientShap(policy)
        # # baseline = torch.randn(*processed_img.shape).to(self.device)  # 选择全零图像作为 baseline
        # #
        # for j in range(K):
        #     saliency = ig.attribute(processed_img, target=j,
        #                                     return_convergence_delta=False).detach().cpu().numpy()
        #     # saliency = gs.attribute(processed_img, baseline, target=j,
        #     #                         return_convergence_delta=False).detach().cpu().numpy()
        #     saliency_maps[j] = saliency.sum(axis=1).squeeze()
        #
        # return saliency_maps, probs.flatten().detach().cpu().numpy()  # (K, H, W)

    def compute_perturbation_saliency(self, policy, processed_img, sigma=3, kernel_size=11, batch_size=4096):
        with torch.no_grad():
            # processed_img = processed_img.clone().to(self.device)
            H, W = processed_img.shape[2], processed_img.shape[3]

            probs_original_logits = policy.compute_learned_prior(processed_img).dist.logits  # (1, K)
            K = probs_original_logits.shape[1]

            saliency_maps = torch.zeros((K, H, W), device=self.device)  # (K, H, W)

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
                    saliency_maps[:, i, j] = delta_probs[idx]

                torch.cuda.empty_cache()

            return saliency_maps.cpu().numpy(), torch.exp(probs_original_logits).flatten().detach().cpu().numpy()

    def compute_instance_influence(self, saliency, instance_masks, percentile=0.99):

        """
        计算每个实例对各个skill的贡献
        参数：
            grads: (K, H, W) 梯度显著性图
            instance_masks: List[(H, W) binary masks]
        返回：
            influence_matrix: (N_instances, 128) 影响矩阵
        """
        influence_matrix = []
        for mask in instance_masks:
            # mask = mask['mask'].squeeze(0)
            mask = mask
            area = mask.sum() + 1e-6
            # 计算每个维度的平均梯度响应
            baseline = saliency.mean(axis=(1, 2), keepdims=True)

            instance_grads = (saliency - baseline) * mask[np.newaxis, :, :]  # (K, H, W)
            influence = instance_grads.sum(axis=(1, 2)) / area
            influence_matrix.append(influence)

        return np.array(influence_matrix)

    # influence_matrix = []
    # for mask in instance_masks:
    #     mask = mask['mask'].squeeze(0)
    #     masked_grads = grads * mask[np.newaxis, :, :]  # (K, H, W)
    #
    #     valid_values = masked_grads[:, mask > 0]  # 仅选择 mask 位置的梯度值
    #     influence = np.percentile(valid_values, percentile, axis=1)
    #
    #     influence_matrix.append(influence)
    #
    # return np.array(influence_matrix)

    def visualize_dimension_influence(self, img, masks, grads, influence_matrix, dist, num_skill, skill_index, hl_step):
        """
        可视化指定维度受实例影响的情况
        """
        # plt.figure(figsize=(12, 4 * num_skill))
        plt.figure(figsize=(16, 4 * 1))

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        for i in range(1):
            # 梯度显著性图
            # plt.subplot(num_skill, 4, 4 * i + 2)
            plt.subplot(1, 4, 4 * i + 2)
            plt.imshow(np.sum(grads * dist[:, np.newaxis, np.newaxis], axis=0), cmap='jet')
            plt.colorbar()
            # plt.title(f"Gradient Map (Skill {i}, prob={dist[i].item():.2f})")
            plt.title(f"Gradient Map (Skill {skill_index}, prob={dist[skill_index].item():.2f})")
            plt.axis('off')

            plt.subplot(1, 4, 4 * i + 3)
            # grad = grads[skill_index]
            grad = np.sum(grads * dist[:, np.newaxis, np.newaxis], axis=0)
            grad_normalized = (grad - grad.min()) / (grad.max() - grad.min())  # 归一化梯度图
            grad_colored = plt.cm.get_cmap('jet')(grad_normalized)
            # grad_colored[:, :, -1] = 0.5  # 设置透明度
            plt.imshow(img)
            plt.imshow(grad_colored, alpha=0.7)  # 叠加梯度图
            plt.title(f"Overlay")
            plt.axis('off')

            # 实例叠加
            # plt.subplot(num_skill, 4, 4 * i + 4)
            plt.subplot(1, 4, 4 * i + 4)
            plt.imshow(img)
            for j, mask in enumerate(masks):
                mask = mask
                mask_colored = img
                mask_colored[mask > 0] = [255, 0, 0]  # 设置mask区域为红色

                # 叠加mask到图像上
                plt.imshow(mask_colored, alpha=0.5)  # 调整alpha值以改变透明度
                # 在mask旁边添加文本标签
                M = cv2.moments(mask.astype(np.uint8))
                cx = int(M["m10"] / M["m00"])  # 中心 x 坐标
                cy = int(M["m01"] / M["m00"])  # 中心 y
                # y, x = coords[0]
                plt.text(cx, cy,
                         f"{self.candidate_labels[j]}: {influence_matrix[j, skill_index] * 1000:.2f}",
                         color='white', fontsize=8,
                         backgroundcolor='black')
            plt.title("Instance Contributions (×0.01)")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.conf.exp_path}/skill_influence_{hl_step}.pdf')

    def get_exp_dir(self):
        return os.environ['EXP_DIR']


if __name__ == '__main__':
    InstanceInfluence(args=get_args())
