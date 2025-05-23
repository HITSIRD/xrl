import imp

from collections import defaultdict
from shutil import copy
import datetime
import random

import numpy as np
import torch
from etils.epath.register import make_path
from torch.utils.tensorboard import SummaryWriter

from src.components.checkpointer import save_cmd, get_config_path, CheckpointHandler
from src.rl.components.params import get_args
from src.rl.components.buffer import RolloutStorage
from src.utils.general import AttrDict, map_dict, AverageMeter, AverageTimer, timing
from src.utils.wandb import WandBLogger
from src.configs.local import *


class RLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conf = conf = self.get_config()
        self.conf.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self._hp = self.conf.general  # override defaults with config file

        self.log_dir = os.path.join(self.conf.exp_path, 'events')
        self.logger = self.get_logger(conf, self.log_dir, wandb=True)
        self.env = self._hp.environment(self.conf.env)
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        self.global_step, start_epoch = 0, 0
        self.n_update_steps = 0
        self.train(self._hp.num_epochs)

    def train(self, num_epochs):
        if self._hp.n_warmup_steps > 0:
            self.warmup()

        timers = defaultdict(lambda: AverageTimer())

        for epoch in range(num_epochs):
            print('starting epoch ', epoch)

            self.sampler.init(is_train=True)
            ep_start_step = self.global_step
            while self.global_step - epoch * self._hp.n_steps_per_epoch < self._hp.n_steps_per_epoch:
                with timers['batch'].time():
                    # collect experience
                    with timers['rollout'].time():
                        experience_batch, env_steps, info = self.sampler.sample_batch(
                            batch_size=self._hp.n_steps_per_update,
                            global_step=self.global_step,
                            store_ll=False)
                        self.global_step += env_steps

                    # update policy
                    with timers['update'].time():
                        agent_outputs = self.agent.update(experience_batch, info)
                        self.n_update_steps += self.agent.update_iterations

                    # log results
                    with timers['log'].time():
                        # if self.log_outputs_now:
                        self.agent.log_outputs(agent_outputs, None, self.logger,
                                               log_images=False, step=self.global_step)
                        self.print_train_update(epoch, agent_outputs, timers)

            save_checkpoint({
                'epoch': epoch,
                'global_step': self.global_step,
                'state_dict': self.agent.state_dict(),
            }, os.path.join(self.conf.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))
            self.agent.save_state(self.conf.exp_path)
            self.val()

    def val(self):
        """Evaluate agent."""
        val_rollout_storage = RolloutStorage()
        with self.agent.val_mode():
            with torch.no_grad():
                with timing("Eval rollout time: "):
                    for _ in range(WandBLogger.N_LOGGED_SAMPLES):  # for efficiency instead of self.args.n_val_samples
                        val_rollout_storage.append(
                            self.sampler.sample_episode(is_train=False, render=False))

        rollout_stats = val_rollout_storage.rollout_stats()
        with timing("Eval log time: "):
            self.agent.log_outputs(rollout_stats, val_rollout_storage,
                                   self.logger, log_images=False, step=self.global_step)
        print("Evaluation Avg_Reward: {}".format(rollout_stats.avg_reward))
        del val_rollout_storage

    def warmup(self):
        print("Warmup data collection for {} steps...".format(self._hp.n_warmup_steps))
        with self.agent.rand_act_mode():
            self.sampler.init(is_train=True)
            warmup_experience_batch, _ = self.sampler.sample_batch(batch_size=int(self._hp.n_warmup_steps),
                                                                   store_ll=False)
        self.agent.add_experience(warmup_experience_batch)
        print("...Warmup done!")

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
            logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                 path=self.conf.exp_path, conf=conf,
                                 exclude=['model_rewards', 'data_dataset_spec_rewards'])
        else:
            logger = None

        return logger

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
                                     path=self.conf.exp_path, conf=conf)
            else:
                logger = SummaryWriter(log_dir)
        else:
            logger = None

        return logger

    def print_train_update(self, epoch, agent_outputs, timers):
        print('Train Epoch: {} [It {}/{} ({:.0f}%)]'.format(
            epoch, self.global_step, self._hp.n_steps_per_epoch * self._hp.num_epochs,
                                     100. * self.global_step / (self._hp.n_steps_per_epoch * self._hp.num_epochs)))
        print('avg time for rollout: {:.3f}s, update: {:.3f}s, logs: {:.3f}s, total: {:.3f}s'
              .format(timers['rollout'].avg, timers['update'].avg, timers['log'].avg,
                      timers['rollout'].avg + timers['update'].avg + timers['log'].avg))
        if hasattr(self.conf.agent, 'hl_interval'):
            interval = self.conf.agent.hl_interval
        else:
            interval = 1.0

        togo_train_time = timers['batch'].avg * (
                self._hp.num_epochs * self._hp.n_steps_per_epoch - self.global_step) / 3600. / interval
        # print('FPS: {}'.format(interval / timers['batch'].avg))
        print('ETA: {:.3f}h'.format(togo_train_time))

    @property
    def log_outputs_now(self):
        return self.global_step % (int(self._hp.n_steps_per_epoch) / self._hp.log_output_per_epoch) == 0


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
    RLTrainer(args=get_args())
