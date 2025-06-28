import datetime

import torch
import os
import imp
import json
import copy

import numpy as np

from src.rl.components.params import get_args
from src.train import set_seeds, make_path, datetime_str, save_config, save_checkpoint
from src.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from src.utils.general import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from src.rl.utils.rollout import RolloutSaver, SERolloutSaver, HPRolloutSaver
from src.rl.components.sampler import Sampler
from src.rl.components.buffer import RolloutStorage

class Evaluator:
    """Deterministic high level policy evaluator."""

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = self.get_config()
        self._hp = self.conf.general  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1: self._hp.seed = args.seed  # override from command line if set
        set_seeds(self._hp.seed)

        self.logger = None

        # build env
        self.conf.env.seed = self._hp.seed
        if 'task_params' in self.conf.env: self.conf.env.task_params.seed = self._hp.seed
        if 'general' in self.conf: self.conf.general.seed = self._hp.seed
        self.env = self._hp.environment(copy.deepcopy(self.conf.env))
        self.conf.agent.env_params = self.env.agent_params  # (optional) set params from env for agent
        pretty_print(self.conf)

        # build agent (that holds actor, critic, exposes update method)
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(self.conf.sampler, self.env, self.agent, self.logger, self._hp.max_rollout_len)

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        self.save_rollout = True
        self.save_evaluation = True

        self.val()


    def val(self):
        """Evaluate agent."""
        stat = {}

        if self.args.save_dir is None:
            self.args.save_dir = self._hp.exp_path
        saver = HPRolloutSaver(self.args.save_dir)

        for i in range(1):
            reward = []
            val_rollout_storage = RolloutStorage()
            with self.agent.val_mode():
                with torch.no_grad():
                    with timing(f"index {i} eval rollout time: "):
                        for j in range(self._hp.n_val_sample):
                            # oracle policy
                            # episode = self.sampler.sample_episode(index=i, is_train=False, render=False, task=True)

                            # deterministic policy
                            # episode = self.sampler.sample_episode(index=i, is_train=False, render=False)

                            # spirl_cl_vq & tree policy
                            episode = self.sampler.sample_episode(is_train=False, render=False)

                            # val_rollout_storage.append(episode, reward_only=True)
                            val_rollout_storage.append(episode)
                            reward.append(np.array(episode.reward).sum())

                            if self.save_rollout:
                                saver.save_rollout(episode)
                                saver.save(f"rollout_{j}", save_interval=1, reset=True)

            episode_reward_mean, episode_reward_std = val_rollout_storage.rollout_stats(std=True)
            complete_task, count = val_rollout_storage.evaluate_task()

            print(reward)

            success_rate = count.copy()
            for k in success_rate.keys():
                success_rate[k] = success_rate[k] / self._hp.n_val_sample
            stat[i] = [complete_task, success_rate]

            print(f"index {i} evaluation Avg_Reward: {episode_reward_mean} ({episode_reward_std})")

        if self.save_evaluation:
            now = datetime.datetime.now()
            formatted_date = now.strftime("%Y%m%d_%H%M%S")

            print('writing skill evaluation result...')
            path = os.path.join(self._hp.exp_path, 'skill_evaluate_' + formatted_date + '.json')
            with open(path, "w") as file:
                json.dump(stat, file)

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

    def print_train_update(self, epoch, agent_outputs, timers):
        print('GPU {}: {}'.format(0 if self.use_cuda else 'none',
                                  self._hp.exp_path))
        print('Train Epoch: {} [It {}/{} ({:.0f}%)]'.format(
            epoch, self.global_step, self._hp.n_steps_per_epoch * self._hp.num_epochs,
                                     100. * self.global_step / (self._hp.n_steps_per_epoch * self._hp.num_epochs)))
        print('avg time for rollout: {:.2f}s, update: {:.2f}s, logs: {:.2f}s, total: {:.2f}s'
              .format(timers['rollout'].avg, timers['update'].avg, timers['log'].avg,
                      timers['rollout'].avg + timers['update'].avg + timers['log'].avg))
        togo_train_time = timers['batch'].avg * (self._hp.num_epochs * self._hp.n_steps_per_epoch - self.global_step) \
                          / self._hp.n_steps_per_update / 3600.
        print('ETA: {:.2f}h'.format(togo_train_time))

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    @property
    def log_outputs_now(self):
        return self.n_update_steps % (int(self._hp.n_steps_per_epoch) / self._hp.log_output_per_epoch) == 0

    @property
    def log_images_now(self):
        return self.n_update_steps % int((self._hp.n_steps_per_epoch / self._hp.n_steps_per_update)
                                         / self._hp.log_images_per_epoch) == 0


if __name__ == '__main__':
    Evaluator(args=get_args())
