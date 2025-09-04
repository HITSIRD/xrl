import datetime

import h5py
import numpy as np
import torch
import os
import imp
import json

from matplotlib import pyplot as plt

from src.rl.components.params import get_args
from src.train import make_path
from src.components.checkpointer import get_config_path
from src.utils.general import AttrDict
from src.rl.components.buffer import RolloutStorage
from openai import OpenAI


class InstanceInfluence:
    """Sets up RL training loop, instantiates all components, runs training."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conf = conf = self.get_config()
        self.conf.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self._hp = self.conf.general

        self.args.save_dir = os.path.join(self.conf.exp_path, 'explanation', self.args.exp_method,
                                          f'episode_{self.args.episode_idx}')
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        self.analyze()

    def analyze(self):
        """Generate rollouts and save to hdf5 files."""
        policy = self.agent.hl_agent.policy.net
        if hasattr(policy, 'reset_hidden_state'):
            policy.reset_hidden_state()

        for step in [0, 50, 100, 150, 200, 250, 300]:
            img = torch.from_numpy(self._load_img(step=step)).unsqueeze(0).to(self.device)

            with self.agent.val_mode():
                with torch.no_grad():
                    print(step)
                    prior = policy.compute_learned_prior(img)
                    print(prior.dist.logits)
                    # print(prior[0].dist.logits)
                    # print(prior[1].dist.logits)
                    # print(prior[2].dist.logits)

    def _load_img(self, size=256, path='fruits_snacks_0.h5', step=0):
        dir = 'src/data/real_kitchen/fruits-snacks-50-v0'
        with h5py.File(os.path.join(dir, path), 'r') as f:
            image = f['rgb'][step]

        fig = plt.figure(frameon=False)
        plt.axis('off')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(image)
        plt.savefig(f'{self.args.save_dir}/original_img_{step}.png')
        plt.show()
        return image.astype(np.float32).transpose(2, 0, 1) / 255 * 2 - 1

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

    def get_exp_dir(self):
        return os.environ['EXP_DIR']


if __name__ == '__main__':
    InstanceInfluence(args=get_args())
