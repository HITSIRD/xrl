import datetime

import cv2
import h5py
import numpy as np
import torch
import os
import imp
import json

from flax.core.nn import embedding
from matplotlib import pyplot as plt
from nltk.app.wordnet_app import explanation

from skilltree.instance_seg_test import visualize_results
from skilltree.rl.components.params import get_args
from skilltree.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir, save_checkpoint
from skilltree.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from skilltree.utils.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from skilltree.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks, mpi_sum, mpi_gather_experience
from skilltree.rl.components.sampler import Sampler
from skilltree.rl.components.replay_buffer import RolloutStorage
from skilltree.utils.decision_tree_util import tree_to_code, load_decision_program
from instance_seg_test import generate_masks_with_sam, initialize_clip, classify_with_clip
from openai import OpenAI

from skilltree.utils.llm_utils import chain, prompt_template


class InstanceInfluence:
    """Sets up RL training loop, instantiates all components, runs training."""

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = self.get_config()
        # update_with_mpi_config(self.conf)   # self.conf.mpi = AttrDict(is_chef=True)
        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1: self._hp.seed = args.seed  # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        self.logger = None

        # build env
        self.conf.env.seed = self._hp.seed
        if 'task_params' in self.conf.env: self.conf.env.task_params.seed = self._hp.seed
        if 'general' in self.conf: self.conf.general.seed = self._hp.seed
        self.env = self._hp.environment(self.conf.env)
        self.conf.agent.env_params = self.env.agent_params  # (optional) set params from env for agent

        # build agent (that holds actor, critic, exposes update method)
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0

        self.candidate_labels = ["microwave",
                                 "slide cabinet",
                                 "hinge cabinet",
                                 "light switch",
                                 "burner switch",
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
        if self.args.save_dir is None:
            self.args.save_dir = self._hp.exp_path
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # sample an episode
        reward = 0

        val_rollout_storage = RolloutStorage()

        while reward <= 2:
            with self.agent.val_mode():
                with torch.no_grad():
                    # oracle policy
                    # episode = self.sampler.sample_episode(index=i, is_train=False, render=False, task=True)

                    # deterministic policy
                    # episode = self.sampler.sample_episode(index=i, is_train=False, render=False, task=False)

                    # spirl_cl_vq & tree policy
                    episode = self.sampler.sample_episode(is_train=False, render=False, task=False)

                    # val_rollout_storage.append(episode, reward_only=True)
                    val_rollout_storage.append(episode)
                    reward = np.array(episode.reward).sum()

                # episode_reward_mean, episode_reward_std = val_rollout_storage.rollout_stats(std=True)
                # complete_task, count = val_rollout_storage.evaluate_task()

                print(reward)

        # file = 'data/kitchen/kitchen-mixed-v0/kitchen-mixed-v0_0.h5'
        # with h5py.File(file, 'r') as dataset:
        #     image_obs = torch.from_numpy(dataset['traj']['observations'][150]).to(self.device).unsqueeze(0)

        # convert tree to code

        feature_names = []

        # kitchen
        for i in range(128):
            feature_names.append(f'x_{i}')
        tree = self.agent.hl_agent.policy.tree
        tree_to_code(tree, feature_names)

        # initialize clip model
        clip_model, clip_processor = initialize_clip(device=self.device)

        # 加载动态生成的决策树函数
        traced_predict = load_decision_program("decision_code.txt")
        encoder = self.agent.hl_agent.policy.net.img_encoder_p

        index = 0

        for i in range(len(episode['observation'])):
            if episode['is_hl_step'][i]:
                obs = episode['observation'][i]
                with torch.no_grad():
                    obs = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                    image_obs = self.agent.hl_agent.policy.net.unflatten_obs(obs).prior_obs
                    obs = encoder(image_obs).cpu().numpy()

                print(f"index {index}")

                prediction, decision_path, path_index = traced_predict(obs.flatten(), feature_names)
                print(f"Prediction: {prediction}")
                print(f"Decision Path: {decision_path}")
                print(f"Path Index: {path_index}")

                influence_str = ""

                with self.agent.val_mode():
                    # with torch.no_grad():
                    # num_samples = image_obs.shape[0]
                    processed_img = image_obs

                    # processed_img = image_obs.permute(0, 3, 1, 2).float()
                    # processed_img = processed_img / 255 * 2 - 1
                    # image_embedding = encoder(processed_img)

                    # Step 0: 获取实例分割结果
                    img = (processed_img * 0.5 + 0.5) * 255.0
                    img = img.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.uint8)
                    self.save_img(img, index)
                    masks = generate_masks_with_sam(img)
                    results = classify_with_clip(clip_model, clip_processor, img, masks, self.candidate_labels)

                    # 计算梯度显著性
                    grads = self.compute_gradient_saliency(encoder, processed_img)  # (128, H, W)
                    influence_matrix = self.compute_instance_influence(grads, results)  # (N, 128)
                    self.visualize_influence(influence_matrix, index)

                    print("influence matrix shape: ", influence_matrix.shape)
                    # print(influence_matrix)

                    self.visualize_dimension_influence(img, results, grads, influence_matrix, path_index, index)

                    for dim in path_index:
                        influence_str += f'x_{dim}\n'
                        for i, (label, influence) in enumerate(zip(self.candidate_labels, influence_matrix[:, dim])):
                            influence_str += f'Object {i} ({label}): {influence:.4f}\n'

                    print(influence_str)

                    explanation = chain.invoke(prompt_template.format(decision_path=decision_path, score=influence_str))
                    print(explanation)
                    self.save_explanation(explanation, index)

                index += 1

    def save_img(self, img, index):
        # image = cv2.imread(img)
        plt.imshow(img)
        plt.savefig(f'original_img_{index}.png')

    def save_explanation(self, explanation, index):
        file_path = f'explanation_{index}.md'

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(explanation)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and agent configs
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

        # load notes if there are any
        if self.args.notes != '':
            conf.notes = self.args.notes
        else:
            try:
                conf.notes = conf_module.notes
            except:
                conf.notes = ''

        # load config overwrites
        if self.args.config_override != '':
            for override in self.args.config_override.split(','):
                key_str, value_str = override.split('=')
                keys = key_str.split('.')
                curr = conf
                for key in keys[:-1]:
                    curr = curr[key]
                curr[keys[-1]] = type(curr[keys[-1]])(value_str)

        return conf

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def visualize_influence(self, matrix, index):
        plt.figure(figsize=(7, 4))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.title('Instance Influence')
        plt.xlabel('Dimensions')
        plt.ylabel('Objects')
        plt.show()
        plt.savefig(f'instance_influence_{index}.pdf')

    def compute_gradient_saliency(self, encoder, img):
        """
        计算输入图像对embedding各维度的梯度显著性
        返回：
            grads: (128, H, W) 的梯度张量，每个维度对应一个空间梯度图
        """
        img.requires_grad_(True)
        embedding = encoder(img)

        grads = []
        for j in range(embedding.shape[1]):
            encoder.zero_grad()
            # 反向传播计算单个维度的梯度
            grad_output = torch.zeros_like(embedding)
            grad_output[0, j] = 1.0  # 仅保留目标维度的梯度
            embedding.backward(gradient=grad_output, retain_graph=True)
            # 提取输入图像的梯度
            grad = img.grad.data.abs().sum(dim=1).squeeze()  # 合并通道维度
            grads.append(grad.cpu().numpy())

        return np.stack(grads, axis=0)  # (128, H, W)

    def compute_instance_influence(self, grads, instance_masks):
        """
        计算每个实例对各个维度的贡献
        参数：
            grads: (128, H, W) 梯度显著性图
            instance_masks: List[(H, W) binary masks]
        返回：
            influence_matrix: (N_instances, 128) 影响矩阵
        """
        influence_matrix = []
        for mask in instance_masks:
            mask = mask['mask'].squeeze(0)
            area = mask.sum() + 1e-6
            # 计算每个维度的平均梯度响应
            instance_grads = grads * mask[np.newaxis, :, :]  # (128, H, W)
            influence = instance_grads.sum(axis=(1, 2)) / area
            influence_matrix.append(influence)

        return np.array(influence_matrix)

    def visualize_dimension_influence(self, img, masks, grads, influence_matrix, dim_idx, index):
        """
        可视化指定维度受实例影响的情况
        """
        path_len = len(dim_idx)
        plt.figure(figsize=(12, 4 * path_len))

        plt.subplot(path_len, 3, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        for i, dim in enumerate(dim_idx):
            # 梯度显著性图
            plt.subplot(path_len, 3, 3 * i + 2)
            plt.imshow(grads[dim], cmap='jet')
            plt.colorbar()
            plt.title(f"Gradient Map (Dim {dim})")
            plt.axis('off')

            # 实例叠加
            plt.subplot(path_len, 3, 3 * i + 3)
            plt.imshow(img)
            for j, mask in enumerate(masks):
                mask = mask['mask'].squeeze(0)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], linewidth=2)
                    plt.text(cnt[0, 0, 0], cnt[0, 0, 1],
                             f"{influence_matrix[j, dim]:.2f}",
                             color='white', fontsize=10,
                             backgroundcolor='black')
            plt.title("Instance Contributions")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'dimension_influence_{index}.pdf')

    @property
    def log_outputs_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                         / self._hp.log_output_per_epoch) == 0 \
            or self.log_images_now

    @property
    def log_images_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                         / self._hp.log_images_per_epoch) == 0

    @property
    def is_chef(self):
        return self.conf.mpi.is_chef

    @property
    def use_multiple_workers(self):
        return self.conf.mpi.num_workers > 1


if __name__ == '__main__':
    InstanceInfluence(args=get_args())
