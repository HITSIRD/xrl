import os

import h5py
from src.components.data_loader import Dataset
import numpy as np
from src.utils.general import AttrDict

# TASKS_DICT = {
#     'open_fridge': 0,
#     'close_fridge': 1,
#     'open_cab': 2,
#     'store_mango': 3,
#     'store_jello': 4,
# }

TASKS_DICT = {
    'open_fridge': 0,
    'close_fridge': 1,
    'open_cab': 2,
    'close_cab': 3,
    'store_mango': 4,
    'store_lemon': 5,
    'store_orange': 6,
    'store_cheezit': 7,
    'store_jello': 8
}


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
        self.skill_enc = np.eye(len(TASKS_DICT))
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
        str2index = np.vectorize(lambda x: TASKS_DICT[x.decode('utf-8')])
        return str2index(skills)
