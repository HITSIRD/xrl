import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from src.modules.networks import CNNEncoder
from src.rl.components.agent import BaseAgent
from src.rl.components.policy import Policy
from src.utils.general import AttrDict, ParamDict


class DQNPolicy(Policy):
    def __init__(self, config):
        self._hp = config
        self.update_model_params(self._hp)
        super().__init__()

        self.epsilon = self._hp.epsilon
        self.eps_decay = self._hp.eps_decay
        self.eps_min = self._hp.eps_min
        self.tau = self._hp.tau

        self.q_eval = DuelingDeepQNetwork(config=self._hp)
        self.q_target = DuelingDeepQNetwork(config=self._hp)

        # self.memory = ReplayBuffer(state_dim=state_dim, task_dim=task_dim, action_dim=action_dim,
        #                            max_size=self.max_size, batch_size=self.batch_size)

        self.update_network_parameters(tau=1.0)
        self.last_q_params = self.q_eval.parameters()
        self.codebook = self._load_codebook()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

        # print('update network parameters')

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min

    def save_parameters(self):
        self.last_q_params = self.q_eval.parameters()

    def _build_network(self):
        self._hp.prior_model_params.device = self._hp.device
        net = self._hp.prior_model(self._hp.prior_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.prior_model_epoch)
        return net

    def recover(self):
        for last_q_params, q_eval_params in zip(self.last_q_params, self.q_eval.parameters()):
            q_eval_params.data.copy_(last_q_params)

    def forward(self, obs):
        obs = self._split_obs(obs)
        q_vals = self.q_eval.forward(obs.images)
        action = T.argmax(q_vals, dim=-1)

        size = action.shape[0] if action.dim() > 0 else 1
        index = torch.arange(size, dtype=torch.int).to(self.device)
        if np.random.random() < self.epsilon:
            action = torch.from_numpy(np.array(np.random.choice(self._hp.skill_dim, size))).int()
        return AttrDict(action=self.codebook[action], action_index=action, value=q_vals[index, action])

    def sample_rand(self, obs, prior=True):
        if prior:
            obs = self._split_obs(obs)
            index = torch.argmax(self.net.prior_head(self.net.prior_encoder(obs.images)), dim=-1)
            return AttrDict(action=self.codebook[index], action_index=index)
        return self.forward(obs)

    def _split_obs(self, obs):
        if isinstance(obs, AttrDict):
            return AttrDict(
                cond_input=self.net.enc_obs(obs.obs),
                z=obs.hl_action,
            )
        else:
            unflattened_obs = self.net.unflatten_obs(obs)
            return AttrDict(
                images=unflattened_obs.prior_obs,
                skills=obs[:, -self.net.latent_dim:],
            )

    def _load_codebook(self):
        return np.eye(self._hp.skill_dim)

    @staticmethod
    def update_model_params(params):
        params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, config):
        super(DuelingDeepQNetwork, self).__init__()
        self.config = config
        self.V, self.A = self._build_net()

        self.encoder = self.build_encoder()

    def forward(self, obs):
        x = obs
        x = self.encoder(x)

        v = self.V(x)
        a = self.A(x)
        q = v + a - T.mean(a, dim=-1, keepdim=True)

        return q

    def build_encoder(self):
        return CNNEncoder(3, self.config.input_res, self.config.img_enc_dim)

    def _build_net(self):
        V = nn.Linear(self.config.img_enc_dim, 1)
        A = nn.Linear(self.config.img_enc_dim, self.config.skill_dim)

        return V, A

# class ReplayBuffer:
#     def __init__(self, state_dim, task_dim, action_dim, max_size, batch_size):
#         self.mem_size = max_size
#         self.batch_size = batch_size
#         self.mem_cnt = 0
#
#         self.state_memory = np.zeros((self.mem_size, *state_dim))
#         self.task_memory = np.zeros((self.mem_size, task_dim))
#         self.action_memory = np.zeros((self.mem_size,))
#         self.reward_memory = np.zeros((self.mem_size,))
#         self.next_state_memory = np.zeros((self.mem_size, *state_dim))
#         self.terminal_memory = np.zeros((self.mem_size,), dtype=bool)
#
#     def insert(self, state, task, action, reward, state_, done):
#         mem_idx = self.mem_cnt % self.mem_size
#
#         self.state_memory[mem_idx] = state.cpu()
#         self.task_memory[mem_idx] = task.cpu()
#         self.action_memory[mem_idx] = action.cpu()
#         self.reward_memory[mem_idx] = reward.cpu()
#         self.next_state_memory[mem_idx] = state_.cpu()
#         self.terminal_memory[mem_idx] = done
#
#         self.mem_cnt += 1
#
#     def sample_buffer(self):
#         mem_len = min(self.mem_size, self.mem_cnt)
#
#         batch = np.random.choice(mem_len, self.batch_size, replace=False)
#
#         states = self.state_memory[batch]
#         tasks = self.task_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         states_ = self.next_state_memory[batch]
#         terminals = self.terminal_memory[batch]
#
#         return states, tasks, actions, rewards, states_, terminals
#
#     def ready(self):
#         return self.mem_cnt > self.batch_size
