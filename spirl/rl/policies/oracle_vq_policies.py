import json

import torch
import numpy as np

from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.rl.components.agent import BaseAgent
from spirl.rl.components.policy import Policy

TASK_ELEMENTS = {'microwave': 0,
                 'kettle': 1,
                 'bottom burner': 2,
                 'top burner': 3,
                 'light switch': 4,
                 'slide cabinet': 5,
                 'hinge cabinet': 6
                 }

N_TASKS = len(TASK_ELEMENTS)

class OracleVQPolicy(Policy):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.update_model_params(self._hp.prior_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None
        # self.task = ['microwave', 'kettle', 'bottom burner', 'light switch']
        with open(self._hp.skill_evaluation, 'r') as f:
            stat = json.load(f)

        self.rate = np.zeros((self._hp.prior_model_params.codebook_K, N_TASKS))
        for z in stat.keys():
            for k, v in stat[z][1].items():
                self.rate[int(z)][TASK_ELEMENTS[k]] = v

        self.sequence_prob = np.zeros(
            (self._hp.prior_model_params.codebook_K, N_TASKS * N_TASKS))  # len * (len - 1)
        for z in stat.keys():
            for tasks in stat[z][0]:
                if len(tasks) > 1:
                    for i in range(len(tasks) - 1):
                        index_0 = TASK_ELEMENTS[tasks[i][0]]
                        index_1 = TASK_ELEMENTS[tasks[i + 1][0]]
                        self.sequence_prob[int(z)][index_0 * N_TASKS + index_1] += 1

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy_model': None,  # policy model class
            'policy_model_params': None,  # parameters for the policy model
            'policy_model_checkpoint': None,  # checkpoint path of the policy model
            'policy_model_epoch': 'latest',  # epoch that checkpoint should be loaded for (defaults to latest)
            'load_weights': True,  # optionally allows to *not* load the weights (ie train from scratch)
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, task):
        if task[0] is None:
            index = TASK_ELEMENTS[task[1]]
            # print(task[1])
            # print(np.argmax(self.rate[:, index]))
            return self.net.codebook.embedding.weight[torch.IntTensor([np.argmax(self.rate[:, index])])]
        else:
            index_0 = TASK_ELEMENTS[task[0]]
            index_1 = TASK_ELEMENTS[task[1]]
            # print(task[1])
            # print(np.argmax(self.rate[:, index_0] * self.rate[:, index_1]))
            prob = self.rate[:, index_0] * self.rate[:, index_1]
            # print(prob)
            # return self.net.codebook.embedding.weight[torch.IntTensor([np.argmax(prob)])]
            return self.net.codebook.embedding.weight[torch.IntTensor([9])]

            index = index_0 * N_TASKS + index_1
            # print(np.argmax(self.sequence_prob[:, index]))
            # print(self.sequence_prob[:, index])
            return self.net.codebook.embedding.weight[torch.IntTensor([np.argmax(self.sequence_prob[:, index])])]

    def _build_network(self):
        net = self._hp.prior_model(self._hp.prior_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.policy_model_epoch)
        return net

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

    def _split_obs(self, obs):
        assert obs.shape[1] == self.net.state_dim + self.net.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self.net.latent_dim],  # condition decoding on state
            z=obs[:, -self.net.latent_dim:],
        )

    @staticmethod
    def update_model_params(params):
        params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        params.batch_size = 1  # run only single-element batches for forward pass

    @property
    def horizon(self):
        return self._hp.policy_model_params.n_rollout_steps
