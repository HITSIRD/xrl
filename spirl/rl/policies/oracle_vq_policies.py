import copy
import json

import pybullet
import torch
import numpy as np

from spirl.utils.general_utils import AttrDict, ParamDict, listdict2dictlist
from spirl.rl.components.agent import BaseAgent
from spirl.rl.components.policy import Policy

# kitchen
# TASK_ELEMENTS = {'microwave': 0,
#                  'kettle': 1,
#                  'bottom burner': 2,
#                  'top burner': 3,
#                  'light switch': 4,
#                  'slide cabinet': 5,
#                  'hinge cabinet': 6
#                  }

# calvin
# TASK_ELEMENTS = {'open_drawer': 0,
#                  'turn_on_lightbulb': 1,
#                  'move_slider_left': 2,
#                  'turn_on_led': 3
#                  }

TASK_ELEMENTS = {'open_drawer': 0,
                 'move_slider_left': 1,
                 'turn_on_led': 2,
                 'turn_on_lightbulb': 3,
                 }

N_TASKS = len(TASK_ELEMENTS)


# class OracleVQPolicy(Policy):
#     def __init__(self, config):
#         self._hp = self._default_hparams().overwrite(config)
#         self.update_model_params(self._hp.prior_model_params)
#         super().__init__()
#         self.copy_env = None
#         self.codebook = self.net.codebook.embedding.weight.cpu().detach().numpy()
#
#     def _default_hparams(self):
#         default_dict = ParamDict({
#             'eval_episode': 1,
#             'eval_steps': 100,
#             'policy_model_epoch': 99,
#         })
#         return super()._default_hparams().overwrite(default_dict)
#
#     def forward(self, obs, env):
#         self.prob = np.zeros((self._hp.prior_model_params.codebook_K, self._hp.eval_episode))
#         eval_env = env.env_stat.env(copy.deepcopy(env.env_stat.conf))
#         # eval_env = copy.deepcopy(env.env)
#
#         for i in range(self._hp.prior_model_params.codebook_K):
#             for j in range(self._hp.eval_episode):
#                 # eval_env.reset()
#                 # cid = eval_env.cid
#                 # eval_env.__dict__ = copy.deepcopy(env.env.__dict__)
#
#                 eval_env._env.sim.set_state(env.env._env.sim.get_state())
#
#                 # eval_env.cid = cid
#                 # eval_env.robot.cid = cid
#                 # eval_env.robot.mixed_ik.cid = cid
#                 # eval_env.scene.cid = cid
#                 # eval_env.cameras.cid = cid
#                 # self._copy_state(eval_env, env.env)
#                 episode = self._sample_episode(obs, eval_env, i)
#
#                 reward = np.array(episode['reward']).sum()
#                 # print(reward)
#                 self.prob[i, j] = reward
#                 # eval_env.__del__()
#                 # print(f'del env cid {cid}')
#
#         eval = self.prob.sum(axis=-1)
#         print(eval)
#         greedy_action_index = np.argmax(eval)
#         print(greedy_action_index)
#         return AttrDict(action=self.codebook[greedy_action_index],
#                         action_index=greedy_action_index)
#
#     def _build_network(self):
#         net = self._hp.prior_model(self._hp.prior_model_params, None)
#         BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint)
#         return net
#
#     def reset(self):
#         self.steps_since_hl, self.last_z = np.Inf, None
#
#     def _sample_episode(self, obs, env, index):
#         episode, done = [], False
#         step = 0
#         _obs = obs
#
#         with env.val_mode():
#             with torch.no_grad():
#                 while not done and step < self._hp.eval_steps:
#                     # perform one rollout step
#                     agent_output = self._sample_action(_obs, index=index)
#                     obs, reward, done, info = env.step(agent_output)
#                     episode.append(AttrDict(
#                         reward=reward,
#                         done=done,
#                     ))
#
#                     # update stored observation
#                     _obs = obs
#                     step += 1
#                     # print(self._episode_step)
#
#         return listdict2dictlist(episode)
#
#     def _sample_action(self, obs, index):
#         obs_input = obs[None] if len(obs.shape) == 1 else obs  # need batch input for agents
#         hl_output = AttrDict()
#
#         # perform step with high-level policy
#         hl_output.action = self._act(index=index)
#         # if len(obs_input.shape) == 2 and len(hl_output.shape) == 1:
#         #     hl_output = hl_output[None]  # add batch dim if necessary
#
#         self.net.decoder.eval()
#         output = self.net.decoder(self._make_ll_obs(obs, hl_output.action)).cpu().numpy().flatten()
#         return output
#
#     def _act(self, index):
#         return self.net.codebook.embedding.weight[torch.IntTensor([index])]
#
#     def _make_ll_obs(self, obs, hl_action):
#         """Creates low-level agent's observation from env observation and HL action."""
#         if isinstance(obs, np.ndarray):
#             obs = torch.from_numpy(obs).to(self._hp.prior_model_params.device).float()
#             if obs.ndim == 1:
#                 obs = obs[None]
#         return torch.cat((obs, hl_action), dim=-1)
#
#     def _copy_state(self, eval_env, base_env):
#         eval_env.tasks_to_complete = copy.deepcopy(base_env.target_tasks)
#         eval_env.completed_tasks = copy.deepcopy(base_env.completed_tasks)
#         eval_env.solved_subtasks = copy.deepcopy(base_env.target_tasks)
#         eval_env._t = copy.deepcopy(base_env._t)
#
#     @staticmethod
#     def update_model_params(params):
#         params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         params.batch_size = 1  # run only single-element batches for forward pass

class OracleVQPolicy(Policy):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None
        self.codebook = self._load_codebook()

        # self.task = ['microwave', 'kettle', 'bottom burner', 'light switch']
        self.eval_reward = self._init_eval(self._hp.skill_evaluation)
        self.num_tasks = 4

        # self.sequence_prob = np.zeros(
        #     (self._hp.prior_model_params.codebook_K, N_TASKS * N_TASKS))  # len * (len - 1)
        # for z in stat.keys():
        #     for tasks in stat[z][0]:
        #         if len(tasks) > 1:
        #             for i in range(len(tasks) - 1):
        #                 index_0 = TASK_ELEMENTS[tasks[i][0]]
        #                 index_1 = TASK_ELEMENTS[tasks[i + 1][0]]
        #                 self.sequence_prob[int(z)][index_0 * N_TASKS + index_1] += 1

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy_model': None,  # policy model class
            'policy_model_params': None,  # parameters for the policy model
            'policy_model_checkpoint': None,  # checkpoint path of the policy model
            'policy_model_epoch': 'latest',  # epoch that checkpoint should be loaded for (defaults to latest)
            'load_weights': True,  # optionally allows to *not* load the weights (ie train from scratch)
        })
        return super()._default_hparams().overwrite(default_dict)

    def _init_eval(self, eval_path):
        eval_reward = []

        for i in range(4):
            try:
                file = eval_path + str(i) + '.json'
                with open(file, 'r') as f:
                    stat = json.load(f)
            except FileNotFoundError:
                continue

            rate = np.zeros((self._hp.prior_model_params.codebook_K, N_TASKS))
            for z in stat.keys():
                for k, v in stat[z][1].items():
                    rate[int(z)][TASK_ELEMENTS[k]] = v

            eval_reward.append(rate.sum(axis=-1))
            # eval_reward.append(rate)

        print(f'load {len(eval_reward)} task evaluation tables of skill')
        return eval_reward

    def forward(self, obs, index, task):
        num_to_complete_tasks = len(task)
        num_eval_task = len(self.eval_reward)

        if num_eval_task > self.num_tasks - num_to_complete_tasks:
            action_index = np.argmax(
                self.eval_reward[self.num_tasks - num_to_complete_tasks])
            # action_index = np.argmax(
            #     self.eval_reward[self.num_tasks - num_to_complete_tasks][self.num_tasks - num_to_complete_tasks])
        else:
            action_index = index

        return AttrDict(action=self.codebook[action_index], action_index=action_index)

        # old version
        # if task[0] is None:
        #     index = TASK_ELEMENTS[task[1]]
        #     # print(task[1])
        #     # print(np.argmax(self.rate[:, index]))
        #     return self.net.codebook.embedding.weight[torch.IntTensor([np.argmax(self.rate[:, index])])]
        # else:
        #     index_0 = TASK_ELEMENTS[task[0]]
        #     index_1 = TASK_ELEMENTS[task[1]]
        #     # print(task[1])
        #     # print(np.argmax(self.rate[:, index_0] * self.rate[:, index_1]))
        #     prob = self.rate[:, index_0] * self.rate[:, index_1]
        #     # print(prob)
        #     return self.net.codebook.embedding.weight[torch.IntTensor([np.argmax(prob)])]
        #     # return self.net.codebook.embedding.weight[torch.IntTensor([9])]
        #
        #     # index = index_0 * N_TASKS + index_1
        #     # print(np.argmax(self.sequence_prob[:, index]))
        #     # print(self.sequence_prob[:, index])
        #     # return self.net.codebook.embedding.weight[torch.IntTensor([np.argmax(self.sequence_prob[:, index])])]

    def _build_network(self):
        pass

    def _load_codebook(self):
        weight = torch.load(self._hp.codebook_checkpoint)
        print('loading codebook from {}'.format(self._hp.codebook_checkpoint))

        # return weight['state_dict']['hl_agent']['policy.prior_net.codebook.embedding.weight'].cpu().numpy()
        # return weight['state_dict']['hl_agent']['policy.net.codebook.embedding.weight'].cpu().numpy()
        return weight['state_dict']['codebook.embedding.weight'].cpu().numpy()

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

    def _split_obs(self, obs):
        assert obs.shape[1] == self.net.state_dim + self.net.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self.net.latent_dim],  # condition decoding on state
            z=obs[:, -self.net.latent_dim:],
        )

    @property
    def horizon(self):
        return self._hp.policy_model_params.n_rollout_steps

    @property
    def has_trainable_params(self):
        """Indicates whether policy has trainable params."""
        return False
