import copy
import datetime
import pickle

import cv2
import h5py
import numpy as np
import torch
import os
import imp

from matplotlib import pyplot as plt

from src.rl.components.params import get_args
from src.rl.utils.rollout import HPRolloutSaver
from src.train import set_seeds, make_path
from src.components.checkpointer import get_config_path
from src.utils.dist import kl_categorical
from src.utils.general import AttrDict, get_depth
from instance_seg_test import generate_masks_with_sam, initialize_clip, classify_with_clip
from openai import OpenAI

from src.utils.image import gaussian_blur_perturb, poisson_gaussian_noise_perturb
from src.utils.llm import chain, prompt_template, get_scene_description
from src.utils.pytorch import no_batchnorm_update
from src.utils.saliency import compute_integrated_gradient_saliency, compute_gradient_saliency, compute_gradient_shap, \
    compute_perturbation_saliency, compute_gradCAM_saliency, compute_deeplift_saliency, compute_deepSHAP_saliency, \
    compute_lrp_saliency, compute_occlusion_saliency, compute_instance_influence, compute_guided_backprop_saliency, \
    compute_input_x_gradient_saliency, compute_lime_saliency
from src.utils.video import create_video_from_pdfs_and_markdowns
from src.utils.render import render_mujoco_object_masks
from src.utils.general import MetricAverageMeter


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

        self.skill_labels = self.conf.data.dataset_spec.skill_labels
        self.objects = self.conf.data.dataset_spec.objects
        self.boxes = self.conf.data.dataset_spec.boxes
        self.skill_obj_map = self.conf.data.dataset_spec.skill_obj_map
        self.multi_skill_obj_map = self.conf.data.dataset_spec.multi_skill_obj_map

        self.metric = MetricAverageMeter()

        self.dynamic_objects = get_depth(self.objects) > 1
        self.lang_exp = self.args.lang_exp

        for i in range(self.args.n_episode):
            self.args.episode_idx = i
            self.init_dir()

            if not self.args.overwrite_cache and self.metric.load_from_cache(self.args.save_dir):
                continue
            else:
                episode = self.sample()
                # create_video_from_pdfs_and_markdowns(num_files=25, pdf_dir=self.args.save_dir, md_dir=self.args.save_dir,
                #                                      output_dir=self.args.save_dir, output_video='explanation.mp4',
                #                                      language_output=True, clean_tmp=True)
                self.analyze(episode)

        print(self.metric.compute())

    def init_dir(self):
        self.args.save_dir = os.path.join(self.conf.exp_path, 'explanation', self.args.exp_method,
                                          f'episode_{self.args.episode_idx}')
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

        policy = self.agent.hl_agent.policy
        hl_step = 0
        history = []
        result = []
        if hasattr(policy.net, 'reset_hidden_state'):
            policy.net.reset_hidden_state()

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

                    if self.args.sam_segmentation:
                        # generate masks with names
                        objects = self.objects
                        boxes = self.boxes
                        if self.dynamic_objects:
                            objects = self.objects[skill_index]
                            boxes = self.boxes[skill_index]

                        masks = generate_masks_with_sam(img, objects, boxes, self.args.save_dir)
                    else:
                        masks = render_mujoco_object_masks(self.env, state.squeeze().cpu().numpy())

                    # results = classify_with_clip(clip_model, clip_processor, img, masks, self.candidate_labels)

                    dist = policy.net(processed_img).flatten().detach().cpu().numpy()
                    _, dist_1, dist_2 = policy.net.compute_learned_prior(processed_img)
                    # skill_index_1, skill_index_2 = dist_1.rsample().item(), dist_2.rsample().item()

                    saliency = self.compute_saliency(policy.net, processed_img, skill_index, original_img=img)
                    saliency = self._normalize_saliency(saliency)
                    influence = compute_instance_influence(saliency, masks)  # (N, K)

                # self.visualize_influence(influence, dist, hl_step, skill_index)
                self.visualize_dimension_influence(img, masks, saliency, influence, dist, skill_index, hl_step)

                influence_str = ""
                if self.dynamic_objects:
                    for i, (object, obj_influence) in enumerate(zip(self.objects[skill_index], influence)):
                        influence_str += f'Object {i} ({object}): {obj_influence:.3f}\n'
                else:
                    for i, (object, obj_influence) in enumerate(zip(self.objects, influence)):
                        influence_str += f'Object {i} ({object}): {obj_influence:.3f}\n'

                # print(influence_str)
                eval = self._evaluate(influence, skill_index)
                result.append(eval)
                self.metric.update(eval)

                if self.lang_exp:
                    scene_description = self.load_language_output(hl_step, "description")
                    if not scene_description:
                        scene_description = get_scene_description(img)
                        print("request scene description...")
                        self.save_language_output(scene_description, hl_step, "description")
                    explanation = chain.invoke(
                        prompt_template.format(scene_description=scene_description, skill_index=skill_index,
                                               score=influence_str, history=history))
                    print(explanation)
                    self.save_language_output(explanation, hl_step, "explanation")

                history.append(skill_index)
                hl_step += 1

        self.save_result_to_cache(result)
        create_video_from_pdfs_and_markdowns(num_files=hl_step, pdf_dir=self.args.save_dir, md_dir=self.args.save_dir,
                                             output_dir=self.args.save_dir, output_video='explanation.mp4',
                                             language_output=self.lang_exp, clean_tmp=True)

    def save_img(self, img, index):
        # image = cv2.imread(img)
        plt.imshow(img)
        plt.savefig(f'{self.args.save_dir}/original_img_{index}.png')
        plt.close()

    def save_language_output(self, output, index, prefix):
        file_path = f'{self.args.save_dir}/{prefix}_{index}.md'

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(output)

    def load_language_output(self, index, prefix):
        file_path = f'{self.args.save_dir}/{prefix}_{index}.md'

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            return False
        except Exception:
            return False

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

        objects = self.objects
        if self.dynamic_objects:
            objects = self.objects[skill_index]
        plt.bar(objects, influence)
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

    def compute_saliency(self, policy, processed_img, skill_idx, original_img=None):
        if self.args.exp_method == 'ig':
            return compute_integrated_gradient_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'gradient':
            return compute_gradient_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'gradient_shap':
            return compute_gradient_shap(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'gaussian_perturbation':
            return compute_perturbation_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'guided_backprop':
            return compute_guided_backprop_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'input_x_gradient':
            return compute_input_x_gradient_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'grad_cam':
            return compute_gradCAM_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'guided_grad_cam':
            return compute_gradCAM_saliency(policy, processed_img, skill_idx, layer=False)
        elif self.args.exp_method == 'deep_lift':
            return compute_deeplift_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'deep_shap':
            return compute_deepSHAP_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'lrp':
            return compute_lrp_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'occlusion':
            return compute_occlusion_saliency(policy, processed_img, skill_idx)
        elif self.args.exp_method == 'lime':
            return compute_lime_saliency(policy, processed_img, skill_idx,
                                         generate_masks_with_sam(original_img, combined=True))
        else:
            raise Exception(f'unsupported explanation method {self.args.exp_method}')

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
        plt.title(f"Saliency")
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

        # === 第5列: 编号 → 名称对照 ===
        plt.subplot(2, 4, 5)
        plt.axis('off')

        y = 1.0
        dy = 1.0 / (len(self.objects) + 1)

        objects = self.objects
        if self.dynamic_objects:
            objects = self.objects[skill_index]

        for j, name in enumerate(objects):
            text = f"{j + 1} {name}"

            plt.text(0, y, text,
                     fontsize=14, color='black',
                     va='top', family='monospace')

            y -= dy

        plt.title("Objects", fontsize=14)

        # === 第6列: 选择技能 ===
        plt.subplot(2, 4, 6)
        plt.axis('off')

        y = 1.0
        dy = 1.0 / (len(self.skill_labels) + 1)

        for j, name in enumerate(self.skill_labels):
            text = f"{j + 1} {name} ({dist[j].item():.2f})"

            if skill_index is not None and j == skill_index:
                plt.text(0, y, text,
                         fontsize=15, weight='bold', color='gold',
                         backgroundcolor='black', va='top', family='monospace')
            else:
                plt.text(0, y, text,
                         fontsize=14, color='black',
                         va='top', family='monospace')

            y -= dy

        plt.title("Skills (Prob)", fontsize=14)

        # === 第7–8列: Top-K Influence 柱状图 ===
        plt.subplot(2, 4, (7, 8))
        influence_arr = np.array(influence)
        top_indices = np.argsort(influence_arr)[::-1][:topk]
        top_values = influence_arr[top_indices]
        top_names = [f"{objects[i]}" for i in top_indices]

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

    def _evaluate(self, influence, skill_idx, topk=3):
        """
        评估 saliency 分布在 top-1 / top-k 情况下对 ground truth 的命中情况

        参数:
            influence: List[float]，每个物体的 saliency 值（顺序与 self.labels 对应）
            skill_idx: int，当前选择的skill index
            task_gt_objects: List[str]，整个任务应涉及的 4 个关键物体名称
            topk: int，考虑 top-k 的范围

        返回:
            dict，包括 top-1 准确率、top-k recall、strict 命中等
        """
        objects = self.objects
        if self.dynamic_objects:
            objects = self.objects[skill_idx]
        assert len(influence) == len(objects), "影响值和标签数量不一致"

        # 获取排序后的索引和对应物体
        indices = sorted(range(len(influence)), key=lambda i: -influence[i])
        # ranked_objects = [self.labels[i] for i in indices]
        # topk_objects = ranked_objects[:topk]

        # Top-1 Acc
        top1_hit = int(objects[indices[0]] == self.skill_obj_map[self.skill_labels[skill_idx]])

        # Top-k Recall：命中几个任务目标物体
        multi_relevant_obj = self.multi_skill_obj_map[self.skill_labels[skill_idx]]
        len_candidates_obj = len(set(multi_relevant_obj) & set(objects))
        selected_objects = [objects[i] for i in indices[:len_candidates_obj]]
        correct_topk = len(set(selected_objects) & set(multi_relevant_obj))
        group_recall = correct_topk / len_candidates_obj

        # Strict Top-k 是否完全覆盖所有任务物体
        # strict_topk = int(set(task_gt_objects).issubset(set(indices[:topk])))

        return {
            "top1": top1_hit,
            f"group_recall": group_recall,
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
