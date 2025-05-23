import numpy as np
import gzip
import pickle
import os
import copy

import torch

from src.utils.general import AttrDict, ParamDict, RecursiveAverageMeter


class ReplayBuffer:
    """Stores arbitrary rollout outputs that are provided by AttrDicts."""

    def __init__(self, hp):
        # TODO upgrade to more efficient (vectorized) implementation of rollout storage
        self._hp = hp
        self._max_capacity = self._hp.capacity
        self._replay_buffer = None
        self._idx = None
        self._size = None  # indicates whether all slots in replay buffer were filled at least once

    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if self._replay_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        self._store(experience_batch, idxs)

        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def _store(self, experience_batch, idxs):
        def recursive_store(buffer, batch, idxs):
            for key in batch:
                value = batch[key]

                if isinstance(value[0], dict):
                    # 解包 list of dicts → dict of lists
                    sub_batch = {}
                    for subkey in value[0]:
                        sub_batch[subkey] = [v[subkey] for v in value]
                    recursive_store(buffer[key], sub_batch, idxs)
                else:
                    # 一条一条地写入，确保 shape 正确
                    for i, idx in enumerate(idxs):
                        buffer[key][idx] = np.array(value[i], dtype=np.float32)

        recursive_store(self._replay_buffer, experience_batch, idxs)

    def sample(self, n_samples, filter=None):
        """Samples n_samples from the rollout_storage. Potentially can filter which fields to return."""
        raise NotImplementedError("Needs to be implemented by child class!")

    def get(self):
        """Returns complete replay buffer."""
        return self._replay_buffer

    def reset(self):
        """Deletes all entries from replay buffer and reinitializes."""
        del self._replay_buffer
        self._replay_buffer, self._idx, self._size = None, None, None

    def _init(self, example_batch):
        """Initializes the replay buffer fields given an example experience batch."""
        self._replay_buffer = AttrDict()

        def recursive_init(buffer, example):
            for key in example:
                element = example[key][0]
                if element is not None:
                    if isinstance(element, AttrDict):
                        buffer[key] = AttrDict()
                        recursive_init(buffer[key], element)
                    else:
                        buffer[key] = np.empty(
                            [int(self._max_capacity)] + list(element.shape),
                            dtype=np.float32
                        )

        recursive_init(self._replay_buffer, example_batch)
        self._idx = 0
        self._size = 0

    def save(self, save_dir):
        """Stores compressed replay buffer to file."""
        if not self._hp.dump_replay: return
        os.makedirs(save_dir, exist_ok=True)
        with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'wb') as f:
            pickle.dump(self._replay_buffer, f)
        np.save(os.path.join(save_dir, "idx_size.npy"), np.array([self._idx, self.size]))

    def load(self, save_dir):
        """Loads replay buffer from compressed disk file."""
        assert self._replay_buffer is None  # cannot overwrite existing replay buffer when loading
        if not self._hp.dump_replay:
            return
        with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'rb') as f:
            self._replay_buffer = pickle.load(f)
        idx_size = np.load(os.path.join(save_dir, "idx_size.npy"))
        self._idx, self._size = int(idx_size[0]), int(idx_size[1])

    @staticmethod
    def _get_n_samples(batch):
        """Retrieves the number of samples in batch."""
        for key in batch:
            return len(batch[key])

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._max_capacity

    def __contains__(self, key):
        return key in self._replay_buffer


class UniformReplayBuffer(ReplayBuffer):
    """Samples batch uniformly from all experience samples in the buffer."""

    def sample(self, n_samples, filter=None):
        assert n_samples <= self.size  # need enough samples in replay buffer
        assert isinstance(self.size, int)  # need integer-valued size
        idxs = np.random.choice(np.arange(self.size), size=n_samples)

        def recursive_sample(buffer_section):
            sampled_section = AttrDict()
            for key in buffer_section:
                if filter is None or key in filter:
                    value = buffer_section[key]
                    if isinstance(value, dict):  # 递归处理嵌套结构
                        sampled_section[key] = recursive_sample(value)
                    else:
                        sampled_section[key] = value[idxs]
            return sampled_section

        sampled_transitions = recursive_sample(self._replay_buffer)
        return sampled_transitions


class RolloutBuffer:
    """Stores arbitrary rollout outputs that are provided by AttrDicts."""

    def __init__(self, hp):
        # TODO upgrade to more efficient (vectorized) implementation of rollout storage
        self._hp = hp
        self._max_capacity = self._hp.capacity
        self._rollout_buffer = None
        self._idx = None
        self._size = None  # indicates whether all slots in replay buffer were filled at least once
        self._gamma = self._hp.discount_factor
        self._gae_lambda = self._hp.gae_lambda

    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if self._rollout_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        self._store(experience_batch, idxs)

        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def _store(self, experience_batch, idxs):
        def recursive_store(buffer, batch, idxs):
            for key in batch:
                value = batch[key]

                if isinstance(value[0], dict):
                    # 解包 list of dicts → dict of lists
                    sub_batch = {}
                    for subkey in value[0]:
                        sub_batch[subkey] = [v[subkey] for v in value]
                    recursive_store(buffer[key], sub_batch, idxs)
                else:
                    for i, idx in enumerate(idxs):
                        buffer[key][idx] = np.array(value[i], dtype=np.float32)

        recursive_store(self._rollout_buffer, experience_batch, idxs)

    def sample(self, n_samples, filter=None):
        assert n_samples <= self.size  # need enough samples in replay buffer
        assert isinstance(self.size, int)  # need integer-valued size
        idxs = np.random.choice(np.arange(self.size), size=n_samples)

        def recursive_sample(buffer_section):
            sampled_section = AttrDict()
            for key in buffer_section:
                if filter is None or key in filter:
                    value = buffer_section[key]
                    if isinstance(value, dict):  # 递归处理嵌套结构
                        sampled_section[key] = recursive_sample(value)
                    else:
                        sampled_section[key] = value[idxs]
            return sampled_section

        sampled_transitions = recursive_sample(self._rollout_buffer)
        return sampled_transitions

    def get(self):
        """Returns complete replay buffer."""
        return self._rollout_buffer

    def reset(self):
        """Deletes all entries from replay buffer and reinitializes."""
        self._idx, self._size = 0, 0

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.detach().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self._size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self._rollout_buffer.episode_start[step + 1]
                next_values = self._rollout_buffer.value[step + 1]
            delta = self._rollout_buffer.reward[step] + self._gamma * next_values * next_non_terminal - \
                    self._rollout_buffer.value[step]
            last_gae_lam = delta + self._gamma * self._gae_lambda * next_non_terminal * last_gae_lam
            self._rollout_buffer.advantage[step] = last_gae_lam
        self._rollout_buffer.return_ = self._rollout_buffer.advantage + self._rollout_buffer.value

    def _init(self, example_batch):
        """Initializes the replay buffer fields given an example experience batch."""
        self._rollout_buffer = AttrDict()

        def recursive_init(buffer, example):
            for key in example:
                element = example[key][0]
                if element is not None:
                    if isinstance(element, AttrDict):
                        buffer[key] = AttrDict()
                        recursive_init(buffer[key], element)
                    else:
                        if hasattr(element, "shape"):
                            shape = list(element.shape)
                        else:
                            shape = []
                        buffer[key] = np.empty(
                            [int(self._max_capacity)] + shape,
                            dtype=np.float32
                        )

        recursive_init(self._rollout_buffer, example_batch)

        self._rollout_buffer.advantage = np.zeros(self._max_capacity, dtype=np.float32)
        self._rollout_buffer.value = np.zeros(self._max_capacity, dtype=np.float32)
        self._rollout_buffer.return_ = np.zeros(self._max_capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    @staticmethod
    def _get_n_samples(batch):
        """Retrieves the number of samples in batch."""
        for key in batch:
            return len(batch[key])

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._max_capacity

    def __contains__(self, key):
        return key in self._rollout_buffer


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""

    def __init__(self):
        self.rollouts = []

    def append(self, rollout, reward_only=False):
        """Adds rollout to storage."""
        if reward_only:
            self.rollouts.append(AttrDict(reward=rollout.reward, info=rollout.info))
        else:
            self.rollouts.append(rollout)
        print(f'rollout {len(self.rollouts)}, reward {np.array(rollout.reward).sum()}')

    def rollout_stats(self, std=False):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts  # rollout storage should not be empty

        if not std:
            stats = RecursiveAverageMeter()
            for rollout in self.rollouts:
                stats.update(AttrDict(
                    avg_reward=np.stack(rollout.reward).sum()
                ))
            return stats.avg
        else:
            episode_rewards = []
            for rollout in self.rollouts:
                episode_rewards.append(np.array(rollout.reward).sum())
            episode_rewards = np.array(episode_rewards)
            return episode_rewards.mean(), episode_rewards.std()

    def evaluate_task(self):
        complete_task = []
        for rollout in self.rollouts:
            tasks = []
            for step, task_info in enumerate(rollout['info']):
                if len(task_info[0]['completed_task']) > 0:
                    tasks.append((task_info[0]['completed_task'], step))
            if len(tasks) > 0:
                complete_task.append(tasks)

        count = {}
        for tasks in complete_task:
            for task in tasks:
                if task[0][0] in count:
                    count[task[0][0]] += 1
                else:
                    count[task[0][0]] = 1
        return complete_task, count

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]
