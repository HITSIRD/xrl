import os

import h5py
from src.components.data_loader import Dataset
import numpy as np
from src.utils.general import AttrDict


class RealKitchenDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        # self.subseq_len = self.spec.subseq_len
        self.device = data_conf.device
        self.n_worker = 0
        self.shuffle = shuffle

        print('loading files from', self.data_dir)
        self.filenames = self._get_filenames()
        self.dataset = self._get_samples_per_file(self.filenames)
        self.skill_enc = np.eye(len(self.spec.TASKS_DICT))
        # self.seqs = []
        self.size = 0

        for data in self.dataset:
            self.size += len(data['actions'])

        # self.n_seqs = len(self.seqs)
        self.dataset_size = self.size if dataset_size == -1 else dataset_size

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.size)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.size)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.size)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.size)
            self.end = self.size

    def _get_filenames(self):
        filenames = self._load_h5_files(self.data_dir)

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        # filenames = shuffle_with_seed(filenames)
        # filenames = self._split_with_percentage(self.spec.split, filenames)
        return filenames

    def _load_h5_files(self, dir):
        filenames = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".h5"):
                    filenames.append(os.path.join(root, file))
        return filenames

    def _get_samples_per_file(self, path):
        samples = []

        for file in path:
            data = AttrDict()
            with h5py.File(file, 'r') as F:
                # Fetch data into a dict
                for name in F.keys():
                    if name in ['q']:
                        data.actions = F[name][()].astype(np.float32)
                    if name == 'skill':
                        data.skills = self._preprocess_skills(F[name][()])
                    if name == 'rgb':
                        pass
                    else:
                        data[name] = F[name][()]

                if 'rgb' in F:
                    data.images = self._preprocess_images(F['rgb'][()])
                else:
                    data.images = np.zeros((data.states.shape[0], 2, 2, 3), dtype=np.uint8)

            samples.append(data)
        return samples

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0  # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['mkbl']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        idx = np.random.randint(0, seq.actions.shape[0] - 1)

        output = AttrDict(
            images=seq.images[idx],
            actions=seq.actions[idx].astype(np.float32),
            skills=self.skill_enc[seq.skills[idx]].astype(np.float32),
            complete=True if idx == seq.actions.shape[0] - 1 else seq.skills[idx] != seq.skills[idx + 1],
            # pad_mask=np.ones((self.subseq_len,)),
        )

        return output

    def _sample_seq(self):
        idx = np.random.randint(0, len(self.dataset))
        return self.dataset[idx]

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.size / self.subseq_len)

    def _preprocess_images(self, images):
        assert images.dtype == np.uint8, 'image need to be uint8!'
        # images = resize_video(images, (self.img_sz, self.img_sz))
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255 * 2 - 1
        assert images.dtype == np.float32, 'image need to be float32!'
        return images

    def _preprocess_skills(self, skills):
        str2index = np.vectorize(lambda x: self.spec.TASKS_DICT[x.decode('utf-8')])
        return str2index(skills)


class MultiStepsRealKitchenDataset(RealKitchenDataset):

    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1):
        super().__init__(data_dir, data_conf, phase, shuffle=shuffle, dataset_size=dataset_size)
        self.n_future = 2
        self._add_future_skills()

    def __getitem__(self, index):
        seq = self._sample_seq()
        idx = np.random.randint(0, seq.actions.shape[0] - 1)

        output = AttrDict(
            images=seq.images[idx],
            actions=seq.actions[idx].astype(np.float32),
            skills=self.skill_enc[seq.skills[idx]].astype(np.float32),
            future_skills=self.skill_enc[seq.next_skills[idx]],  # shape [n_future, num_skills]
        )

        return output

    def _add_future_skills(self):
        """为每个 sequence 增加 future skills 信息"""
        assert self.n_future >= 1

        for data in self.dataset:
            skills = data.skills  # shape [T]
            T = len(skills)

            # 存储未来 n_future 个技能
            next_skills = np.full((T, self.n_future), self.spec.TASKS_DICT['end'], dtype=np.int64)

            for i in range(T):
                current = skills[i]
                future = []
                j = i + 1
                # 找到后续变化的技能
                while j < T and len(future) < self.n_future:
                    if skills[j] != skills[j - 1]:
                        future.append(skills[j])
                    j += 1

                # 如果没找到够的，就用 'end' 填充
                while len(future) < self.n_future:
                    future.append(self.spec.TASKS_DICT['end'])

                next_skills[i] = future

            data.next_skills = next_skills  # shape [T, n_future]


class SequenceRealKitchenDataset(RealKitchenDataset):
    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1):
        super().__init__(data_dir, data_conf, phase, shuffle=shuffle, dataset_size=dataset_size)
        self.max_seq_len = data_conf.dataset_spec.max_seq_len
        self.segments = []

        for traj in self.dataset:
            self.segments.append(self._split_by_skill(traj['skills']))

    def _split_by_skill(self, skills):
        segments = []
        prev_skill = skills[0]
        start = 0
        for t in range(1, len(skills)):
            if skills[t] != prev_skill:
                segments.append((prev_skill, start, t))  # skill, start_idx, end_idx
                prev_skill = skills[t]
                start = t
        segments.append((prev_skill, start, len(skills)))
        return segments

    def __getitem__(self, index):
        traj_idx = np.random.randint(len(self.dataset))
        traj = self.dataset[traj_idx]
        images, actions, skills = traj["images"], traj["actions"], traj["skills"]

        segments = self.segments[traj_idx]
        skill_imgs, skill_actions, skill_labels = [], [], []

        idx = np.random.randint(0, segments[0][2] - segments[0][1])
        for (skill, start, end) in segments:
            skill_imgs.append(images[start + idx])  # (C,H,W)
            skill_actions.append(actions[start + idx])  # (C,H,W)
            skill_labels.append(skill)

        start_idx = np.random.randint(0, len(skill_labels) - 2)
        skill_imgs = skill_imgs[start_idx:]
        skill_labels = skill_labels[start_idx:]
        skill_actions = skill_actions[start_idx:]

        if len(skill_labels) >= self.max_seq_len:
            skill_imgs = skill_imgs[:self.max_seq_len]
            skill_labels = skill_labels[:self.max_seq_len]
            skill_actions = skill_actions[:self.max_seq_len]
            pad_mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - len(skill_labels)
            pad_mask = np.array([1] * len(skill_imgs) + [0] * pad_len, dtype=np.float32)
            skill_imgs.extend([np.zeros(skill_imgs[0].shape, dtype=np.float32)] * pad_len)
            skill_actions.extend([np.zeros(skill_actions[0].shape, dtype=np.float32)] * pad_len)
            skill_labels.extend([self.spec.TASKS_DICT['end']] * pad_len)

        skill_imgs = np.stack(skill_imgs)  # (max_seq_len, C, H, W)
        skill_actions = np.stack(skill_actions)
        skill_labels = self.skill_enc[np.array(skill_labels)]  # (max_seq_len, num_classes)

        return AttrDict(
            images=skill_imgs,
            actions=skill_actions.astype(np.float32),
            skills=skill_labels.astype(np.float32),
            pad_mask=pad_mask
        )


class MultiStepsSequenceRealKitchenDataset(SequenceRealKitchenDataset, MultiStepsRealKitchenDataset):

    def __getitem__(self, index):
        traj_idx = np.random.randint(len(self.dataset))
        traj = self.dataset[traj_idx]
        images, actions, skills = traj["images"], traj["actions"], traj["skills"]

        segments = self.segments[traj_idx]
        skill_imgs, skill_actions, skill_labels = [], [], []

        idx = np.random.randint(0, segments[0][2] - segments[0][1])
        for (skill, start, end) in segments:
            skill_imgs.append(images[start + idx])  # (C,H,W)
            skill_actions.append(actions[start + idx])  # (C,H,W)
            skill_labels.append(skill)

        start_idx = np.random.randint(0, len(skill_labels) - self.n_future)
        skill_imgs = skill_imgs[start_idx:]
        skill_labels = skill_labels[start_idx:]
        skill_actions = skill_actions[start_idx:]

        # future skill list
        future_skills_list = []
        for step in range(start_idx + 1, start_idx + self.n_future + 1):
            future_skills = skill_labels[step:]
            future_skills_list.append(future_skills)

        if len(skill_labels) >= self.max_seq_len:
            skill_imgs = skill_imgs[:self.max_seq_len]
            skill_labels = skill_labels[:self.max_seq_len]
            skill_actions = skill_actions[:self.max_seq_len]
            pad_mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - len(skill_labels)
            pad_mask = np.array([1] * len(skill_imgs) + [0] * pad_len, dtype=np.float32)
            skill_imgs.extend([np.zeros(skill_imgs[0].shape, dtype=np.float32)] * pad_len)
            skill_actions.extend([np.zeros(skill_actions[0].shape, dtype=np.float32)] * pad_len)
            skill_labels.extend([self.spec.TASKS_DICT['end']] * pad_len)

        for i in range(len(future_skills_list)):
            if len(future_skills_list[i]) >= self.max_seq_len:
                future_skills_list[i] = future_skills_list[i][:self.max_seq_len]
            else:
                future_skills_list[i].extend(
                    [self.spec.TASKS_DICT['end']] * (self.max_seq_len - len(future_skills_list[i])))

        skill_imgs = np.stack(skill_imgs)  # (max_seq_len, C, H, W)
        skill_actions = np.stack(skill_actions)
        skill_labels = self.skill_enc[np.array(skill_labels)]  # (max_seq_len, num_classes)
        future_skills_array = np.stack([
            self.skill_enc[np.array(future_skills)]
            for future_skills in future_skills_list
        ])

        return AttrDict(
            images=skill_imgs,
            actions=skill_actions.astype(np.float32),
            skills=skill_labels.astype(np.float32),
            future_skills=future_skills_array.astype(np.float32),
            pad_mask=pad_mask
        )
