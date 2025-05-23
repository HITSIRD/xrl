import imp
import os
import time
from shutil import copy
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from etils.epath.register import make_path
from torch.utils.tensorboard import SummaryWriter

from src.components.checkpointer import save_cmd, get_config_path, CheckpointHandler
from src.components.params import get_args
from src.data.kitchen.kitchen_dataloader import KitchenDataset
from src.utils.general import AttrDict, map_dict, AverageMeter
from src.utils.wandb import WandBLogger
from src.configs.local import *

torch.multiprocessing.set_sharing_strategy('file_system')


class SkillTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conf = conf = self.get_config()
        self.conf.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = os.path.join(self.conf.exp_path, 'events')
        self.logger = self.get_logger(conf, self.log_dir, wandb=True)

        self.model = self.conf.general.model(self.conf.model, self.logger).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=conf.general.lr)
        self.dataloader = self.get_dataloader(self.args, self.conf.data, phase='train',
                                              n_repeat=conf.general.epoch_cycles_train)

        self.global_step, start_epoch = 0, 0
        self.train(self.conf.general.num_epochs)

    def train(self, num_epochs):
        self.model.train()
        epoch_len = len(self.dataloader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)

        for epoch in range(num_epochs):
            print('starting epoch ', epoch)

            for self.batch_idx, batch in enumerate(self.dataloader):
                data_load_time.update(time.time() - end)
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), batch))

                output = self.model(inputs)
                losses = self.model.loss(output, inputs)

                self.optimizer.zero_grad()
                losses.total.backward()
                self.optimizer.step()

                upto_log_time.update(time.time() - end)

                if self.log_outputs_now:
                    self.model.log_outputs(output, inputs, losses, self.global_step,
                                           log_images=False, phase='train')

                batch_time.update(time.time() - end)
                end = time.time()

                if self.log_outputs_now:
                    # print('GPU {}: {}'.format(0, self._hp.exp_path))
                    print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.global_step, epoch, self.batch_idx, len(self.dataloader),
                        100. * self.batch_idx / len(self.dataloader), losses.total.item())))

                    print('avg time for loading: {:.3f}s, logs: {:.3f}s, compute: {:.3f}s, total: {:.3f}s'
                          .format(data_load_time.avg,
                                  batch_time.avg - upto_log_time.avg,
                                  upto_log_time.avg - data_load_time.avg,
                                  batch_time.avg))
                    togo_train_time = batch_time.avg * (num_epochs - epoch) * epoch_len / 3600.
                    print('BPS: {:.3f}'.format(1. / batch_time.avg))
                    print('ETA: {:.3f}h'.format(togo_train_time))

                self.global_step = self.global_step + 1

            save_checkpoint({
                'epoch': epoch,
                'global_step': self.global_step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, os.path.join(self.conf.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))

    def get_dataloader(self, args, data_conf, phase, n_repeat):
        loader = self.conf.data.dataset_spec.dataset_class(self.conf.general.data_dir, data_conf,
                                                           phase=phase, shuffle=phase == "train"). \
            get_data_loader(self.conf.general.batch_size, n_repeat)

        return loader

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    def get_logger(self, conf, log_dir, wandb=True):
        print('Writing to the experiment directory: {}'.format(self.conf.exp_path))
        if not os.path.exists(self.conf.exp_path):
            os.makedirs(self.conf.exp_path)
        save_cmd(self.conf.exp_path)
        save_config(conf.conf_path, os.path.join(self.conf.exp_path, "conf_" + datetime_str() + ".py"))
        if wandb:
            exp_name = f"{'_'.join(self.args.path.split('/')[-3:])}_{self.args.prefix}" if self.args.prefix \
                else os.path.basename(self.args.path)
            self.b_logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                        path=self.conf.exp_path, conf=conf,
                                        exclude=['model_rewards', 'data_dataset_spec_rewards'])
            writer = self.b_logger
        else:
            writer = SummaryWriter(log_dir)

        return writer

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.model = conf_module.model_config

        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source('dataset_spec',
                                             os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
            data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
        conf.data = data_conf
        conf.model.device = conf.data.device = self.device.type

        # model loading config
        conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self.conf.exp_path))
            if not os.path.exists(self.conf.exp_path):
                os.makedirs(self.conf.exp_path)
            save_cmd(self.conf.exp_path)
            save_config(conf.conf_path, os.path.join(self.conf.exp_path, "conf_" + datetime_str() + ".py"))
            if self.conf.logging_target == 'wandb':
                exp_name = f"{'_'.join(self.args.path.split('/')[-3:])}_{self.args.prefix}" if self.args.prefix \
                    else os.path.basename(self.args.path)
                logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                     path=self.conf.exp_path, conf=conf,
                                     exclude=['model_rewards', 'data_dataset_spec_rewards'])
            else:
                logger = SummaryWriter(log_dir)
        else:
            logger = None

        return logger

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('configs/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)


if __name__ == '__main__':
    SkillTrainer(args=get_args())
