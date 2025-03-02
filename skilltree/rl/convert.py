import datetime

import h5py
import numpy as np
import torch
import os
import imp
import json

from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import defaultdict

from skilltree.rl.components.params import get_args
from skilltree.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir, save_checkpoint
from skilltree.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from skilltree.utils.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from skilltree.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks, mpi_sum, mpi_gather_experience
from skilltree.rl.utils.wandb import WandBLogger
from skilltree.rl.utils.rollout_utils import HPRolloutSaver
from skilltree.rl.components.sampler import Sampler
from skilltree.rl.components.replay_buffer import RolloutStorage


class Collector:
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
        if args.seed != -1: self._hp.seed = args.seed   # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        self.logger = None

        # build env
        self.conf.env.seed = self._hp.seed
        if 'task_params' in self.conf.env: self.conf.env.task_params.seed=self._hp.seed
        if 'general' in self.conf: self.conf.general.seed=self._hp.seed
        self.env = self._hp.environment(self.conf.env)
        self.conf.agent.env_params = self.env.agent_params      # (optional) set params from env for agent

        # build agent (that holds actor, critic, exposes update method)
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        if args.resume or self.conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, self.conf.ckpt_path)
            self._hp.n_warmup_steps = 0     # no warmup if we reload from checkpoint!

        # start training/evaluation
        self.convert()

    def _default_hparams(self):
        default_dict = ParamDict({
            'seed': None,
            'agent': None,
            'data_dir': None,  # directory where dataset is in
            'sampler': Sampler,     # sampler type used
            'exp_path': None,  # Path to the folder with experiments
            'dataset_path': 'experiments/hrl/kitchen/cdt_cl_vq_prior_cdt/mkbl_d6_s1_avgprob/fine_500_50.h5',
        })
        return default_dict

    def convert(self):
        """Generate rollouts and save to hdf5 files."""
        if self.args.save_dir is None:
            self.args.save_dir = self._hp.exp_path
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        file = self._hp.dataset_path
        save_path = os.path.join(self.args.save_dir, "encoded_fine_500_50.h5")

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        batch_size = 1024

        with ((self.agent.val_mode())):
            with torch.no_grad():
                with h5py.File(file, 'r') as dataset:

                    image_obs = torch.from_numpy(np.array(dataset['traj']['states'])).to(self.device)
                    num_samples = image_obs.shape[0]
                    image_embeddings = []

                    for start_idx in range(0, num_samples, batch_size):
                        end_idx = min(start_idx + batch_size, num_samples)
                        batch = self.agent.hl_agent.policy.net.unflatten_obs(image_obs[start_idx:end_idx]).prior_obs
                        image_embeddings.append(self.agent.hl_agent.policy.net.img_encoder_p(batch).cpu())

                    image_embeddings = torch.cat(image_embeddings, dim=0).numpy()
                    traj_data = f.create_group("traj")
                    traj_data.create_dataset("states", data=image_embeddings)
                    traj_data.create_dataset("hl_action_index", data=dataset['traj']['hl_action_index'])

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
        conf.env.device = self.device       # add device to env config as it directly returns tensors

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

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf.conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))

            # setup logger
            logger = None
            if self.args.mode == 'train':
                exp_name = f"{os.path.basename(self.args.path)}_{self.args.prefix}" if self.args.prefix \
                    else os.path.basename(self.args.path)
                if self._hp.logging_target == 'wandb':
                    logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                         path=self._hp.exp_path, conf=conf)
                else:
                    logger = SummaryWriter(log_dir)
            return logger

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        # TODO(karl): check whether that actually loads the optimizer too
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.agent,
                                           load_step=True, strict=self.args.strict_weight_loading)
        self.agent.load_state(self._hp.exp_path)
        self.agent.to(self.device)
        return start_epoch

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
    Collector(args=get_args())