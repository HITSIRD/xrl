import json

import torch
import numpy as np

from skilltree.rl.policies.prior_policies import ACLearnedVQPriorAugmentedPICDTPolicy
from skilltree.utils.general_utils import AttrDict, ParamDict
from skilltree.rl.components.agent import BaseAgent
from skilltree.rl.components.policy import Policy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

import pickle


class CARTPolicy(Policy):
    def __init__(self, config):
        super().__init__(config)
        self._hp = self._default_hparams().overwrite(config)
        self.tree = self._load_tree()
        if self._hp.is_index:
            self.codebook = self._load_codebook()

        # if self._hp.load_img_encoder:
        #     self.img_encoder = self._load_img_encoder()

    def _default_hparams(self):
        default_dict = ParamDict({
            'load_weights': True,  # optionally allows to *not* load the weights (ie train from scratch)
            'max_depth': 9,
            'is_index': True,
            'type': 'classifier',
            'load_img_encoder': True,
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs):
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if self._hp.is_index:
            action_index = self.tree.predict(obs)
            return AttrDict(action=self.codebook[action_index], action_index=action_index)
        else:
            return AttrDict(action=self.tree.predict(obs))

    def update(self, experience):
        self.tree.fit(experience['observation'], experience['hl_action_index'])

        info = AttrDict(
            leaf=self.tree.get_n_leaves(),
            depth=self.tree.get_depth(),
            node_count=self.tree.tree_.node_count,
            # importances=self.tree.feature_importances_,
            # gini=self.tree.tree_.impurity,
            test_score=self.tree.score(experience['observation'], experience['hl_action_index']),
        )
        return info

    def _build_network(self):
        pass

    def _load_tree(self):
        if self._hp.load_weights:
            with open(self._hp.dt_model_checkpoint, 'rb') as f:
                print('loading weights from {}'.format(self._hp.dt_model_checkpoint))
                tree = pickle.load(f)
            return tree

    def _load_codebook(self):
        weight = torch.load(self._hp.codebook_checkpoint)
        print('loading codebook from {}'.format(self._hp.codebook_checkpoint))

        # return weight['state_dict']['hl_agent']['policy.prior_net.codebook.embedding.weight']
        return weight['state_dict']['hl_agent']['policy.net.codebook.embedding.weight']

    # def _load_img_encoder(self):


    @property
    def has_trainable_params(self):
        """Indicates whether policy has trainable params."""
        return False

class ImageCARTPolicy(CARTPolicy, ACLearnedVQPriorAugmentedPICDTPolicy):
    def __init__(self, config):
        # self.net = self._hp.prior_model(self._hp.prior_model_params, None)
        super().__init__(config)
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        default_dict = ParamDict({
            'load_img_encoder': True,
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs):
        if self._hp.load_img_encoder:
            with torch.no_grad():
                img_obs = self.net.unflatten_obs(obs).prior_obs
                obs = self.net.img_encoder(img_obs).cpu().numpy()

        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if self._hp.is_index:
            action_index = self.tree.predict(obs)
            return AttrDict(action=self.codebook[action_index], action_index=action_index)
        else:
            return AttrDict(action=self.tree.predict(obs))

    def _build_network(self):
        if self._hp.policy_model is not None:
            net = self._hp.policy_model(self._hp.policy_model_params, None)
        else:
            net = self._hp.prior_model(self._hp.prior_model_params, None)

        if self._hp.load_weights:
            if self._hp.policy_model is not None:
                BaseAgent.load_model_weights(net, self._hp.policy_model_checkpoint, self._hp.prior_model_epoch)
            else:
                BaseAgent.load_model_weights(net, self._hp.prior_model_checkpoint, self._hp.prior_model_epoch)
        return net
