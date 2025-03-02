import argparse
import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import random
import h5py
import pickle
import configs.local

# logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--dataset_size", default=1000, type=int)

    return parser.parse_args()

class tree_classifier():
    def __init__(self, max_depth=5):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, criterion='gini')

    def train(self, X, Y):
        self.clf.fit(X, Y)

class tree_regressor():
    def __init__(self, max_depth=5):
        self.clf = DecisionTreeRegressor(max_depth=max_depth, criterion='squared_error')

    def train(self, X, Y):
        self.clf.fit(X, Y)


if __name__ == '__main__':
    args = get_args()

    dataset_size = args.dataset_size
    max_depth = args.depth
    env = 'mkbl'

    file_path = os.path.join(f'skilltree/experiments/hrl/kitchen/cdt_cl_vq_prior_cdt/mkbl_d6_s1_avgprob/encoded_fine_500_{dataset_size}.h5')
    file = h5py.File(file_path, 'r')

    dataset = file['traj']
    state = dataset['states'][:]
    hl_action_index = dataset['hl_action_index'][:]

    tree = tree_classifier(max_depth=max_depth)

    tree.train(state, hl_action_index)

    print(f'leaf: {tree.clf.get_n_leaves()}')
    print(f'depth: {tree.clf.get_depth()}')
    print(f'node_count: {tree.clf.tree_.node_count}')
    print(f'importance: {tree.clf.feature_importances_}')
    # print(f'gini: {tree.clf.tree_.impurity}')
    print(f'test_score: {tree.clf.score(state, hl_action_index)}')
    # print(f'test_score: {tree.clf.score(state, action)}')

    model_name = os.path.join(args.path, env, f'cart_fine_{dataset_size}_d{max_depth}.pkl')

    with open(model_name, 'wb') as f:
        print(f'write to {model_name}')
        pickle.dump(tree.clf, f)