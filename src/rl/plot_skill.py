import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

TASK_ELEMENTS = {"bottom burner": 0,
                 "top burner": 1,
                 "light switch": 2,
                 "slide cabinet": 3,
                 "hinge cabinet": 4,
                 "microwave": 5,
                 "kettle": 6
                 }

R_TASK_ELEMENTS = {value: key for key, value in TASK_ELEMENTS.items()}

# TASK_ELEMENTS = {'open_drawer': 0,
#                  'turn_on_lightbulb': 1,
#                  'move_slider_left': 2,
#                  'turn_on_led': 3
#                  }

# TASK_ELEMENTS = {'open_drawer': 0,
#                  'move_slider_left': 1,
#                  'turn_on_led': 2,
#                  'turn_on_lightbulb': 3,
#                  }

K = 7

from scipy.stats import wilcoxon
import nltk
from nltk.metrics import edit_distance


def plot_skill(file_path):
    with open(file_path, 'r') as f:
        stat = json.load(f)

    rate = np.zeros((K, len(TASK_ELEMENTS)))
    for z in stat.keys():
        for k, v in stat[z][1].items():
            rate[int(z)][TASK_ELEMENTS[k]] = v

    plt.figure(figsize=(4.3, 1.6))
    label_size = 6

    x_labels = [str(i) for i in range(K)]
    y_labels = TASK_ELEMENTS.keys()

    plt.xticks(np.arange(K), labels=x_labels, rotation_mode="anchor", ha="center", size=label_size)
    plt.yticks(np.arange(len(TASK_ELEMENTS)), labels=y_labels, size=label_size)
    # plt.title("Skill Evaluation", size=label_size)

    # plt.axis.xaxis.

    for i in range(K):
        for j in range(len(TASK_ELEMENTS)):
            if rate[i, j] > 0.0001:
                plt.text(i, j, rate[i, j], ha="center", va="center", color="w", size=label_size - 2)
            if i == K - 1 and np.sum(rate[:, j]) > 0.0001:
                plt.text(K, j, np.around(np.sum(rate[:, j]), 2), ha="center", va="center", color="b",
                         size=label_size - 2)
    print(np.sum(rate))

    plt.imshow(rate.T)
    cb = plt.colorbar(fraction=0.0204, pad=0.05)
    cb.ax.tick_params(labelsize=label_size)
    plt.tight_layout(pad=0.2)
    plt.savefig(file_path + '.pdf')


def plot_task_transition(file_path, aggregate=False):
    with open(file_path, 'r') as f:
        stat = json.load(f)

    transition = np.zeros((K, len(TASK_ELEMENTS) + 1, len(TASK_ELEMENTS) + 1))
    for index in stat.keys():
        seqs = stat[index][0]
        for i, seq in enumerate(seqs):
            if len(seq) == 0:
                transition[int(index)][0][-1] += 1
            else:
                if len(seq) == 1:
                    transition[int(index)][0][TASK_ELEMENTS[seq[0][0][0]]] += 1
                    transition[int(index)][TASK_ELEMENTS[seq[0][0][0]] + 1][-1] += 1
                else:
                    transition[int(index)][0][TASK_ELEMENTS[seq[0][0][0]]] += 1
                    for j in range(len(seq) - 1):
                        transition[int(index)][TASK_ELEMENTS[seq[j][0][0]] + 1][TASK_ELEMENTS[seq[j + 1][0][0]]] += 1
                    transition[int(index)][TASK_ELEMENTS[seq[-1][0][0]] + 1][-1] += 1

    # normalize the transition prob
    for i in range(transition.shape[1]):
        for j in range(transition.shape[2]):
            weight = np.sum(transition[:, i, j])
            transition[:, i, j] = transition[:, i, j] / 100
            # weight = np.max(transition[:, i, j])
            # print(weight)
            # if weight > 0:
            #     transition[:, i, j] = transition[:, i, j] / weight

    for i in range(transition.shape[0]):
        for j in range(transition.shape[1]):
            weight = np.sum(transition[i, j])
            if weight > 0:
                transition[i, j] = transition[i, j] / weight

    label_size = 8
    x_labels = list(TASK_ELEMENTS.keys()) + ['not finished']
    y_labels = ['start'] + list(TASK_ELEMENTS.keys())

    if not aggregate:
        for i in range(K):
            plt.figure(figsize=(2.4, 2.2))
            plt.xticks(np.arange(len(TASK_ELEMENTS) + 1), labels=x_labels, rotation_mode="anchor", ha="right",
                       rotation=45,
                       size=label_size)
            plt.yticks(np.arange(len(TASK_ELEMENTS) + 1), labels=y_labels, size=label_size)
            plt.imshow(transition[i])
            # cb = plt.colorbar(fraction=0.0204, pad=0.05)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=label_size)
            plt.tight_layout()
            plt.title(f'Skill {i}', size=label_size)
            plt.savefig(f'{file_path}_trans_{i}.pdf')
    else:
        fig, axes = plt.subplots(4, 4, figsize=(8, 7.5))
        for i, ax in enumerate(axes.flat):
            cax = ax.imshow(transition[i])

            ax.set_xticks(np.arange(len(TASK_ELEMENTS) + 1), labels=x_labels, rotation_mode="anchor", ha="right",
                          rotation=45,
                          size=label_size)
            ax.set_yticks(np.arange(len(TASK_ELEMENTS) + 1), labels=y_labels, size=label_size)
            ax.set_title(f'Skill {i}', size=label_size)

            if i % 4 != 0:
                ax.set_yticks([])
            if i < 12:
                ax.set_xticks([])

        ax = fig.add_axes([0.92, 0.35, 0.02, 0.3])
        cb = plt.colorbar(cax, cax=ax)
        cb.set_label('Transition Probability', size=label_size)
        cb.ax.tick_params(labelsize=label_size)
        plt.subplots_adjust(top=0.89, bottom=0.11, left=0.11, right=0.87, wspace=0.08, hspace=0.15)
        plt.suptitle('Kitchen MKBL Subtask Transition', fontsize=label_size + 2)
        plt.savefig(f'{file_path}_trans_all.pdf')


K = 7

# plt.rcParams['text.usetex'] = True

N = 256
vals = np.ones((N, 3))
vals[:, 0] = np.linspace(204 / (N - 1), 247 / (N - 1), N)  # red
vals[:, 1] = np.linspace(231 / (N - 1), 202 / (N - 1), N)  # green
vals[:, 2] = np.linspace(255 / (N - 1), 57 / (N - 1), N)  # blue
cmap = ListedColormap(vals)


def get_filenames_recursive(directory):
    filenames = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            filenames.append(path)
        elif os.path.isdir(path):
            filenames.extend(get_filenames_recursive(path))
    return filenames


def plot_episode(file_path):
    dataset = h5py.File(file_path, 'r')

    skill = np.zeros((len(dataset['traj']['hl_action_index']), K))
    for i, s in enumerate(dataset['traj']['hl_action_index']):
        skill[i, s] = 1

    plt.figure(figsize=(4, 1.6))
    label_size = 5

    interval = 1

    x_labels = [i for i in range(0, len(dataset['traj']['hl_action_index']), interval)]
    y_labels = [i for i in range(K)]

    plt.yticks(np.arange(K), size=label_size)
    plt.xticks(np.arange(len(x_labels)) * interval, labels=x_labels, size=label_size)
    # plt.title("Skill Evaluation", size=label_size)

    # for i in range(K):
    #     for j in range(len(TASK_ELEMENTS)):
    #         plt.text(i, j, rate[i, j], ha="center", va="center", color="w", size=label_size)

    l_s = -0.5
    for i, step in enumerate(dataset['traj']['ct_step']):
        plt.plot([step / 10, step / 10], [-0.5, K - 0.5], color='orange', linewidth=0.5)
        plt.text((step + l_s) / 2 / 10, -1.2,
                 str(dataset['traj']['complete_task'][i][0])[2:-1], size=8, ha='center', va='center')
        l_s = step

    plt.xlabel('Time Step (× 10)', size=8)
    plt.ylabel('Skill Index', size=8)
    plt.imshow(skill.T, aspect='auto', cmap=cmap)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=label_size)
    plt.tight_layout(pad=0.01)
    # plt.title('')
    # plt.show()
    plt.savefig(file_path + '.pdf')


def wilcoxon_test(stat, i, j):
    distance = []
    traj_A = []
    traj_B = []
    for traj in stat[str(i)][0]:
        st = []
        for task in traj:
            st.append(TASK_ELEMENTS[task[0][0]])
        traj_A.append(st)

    for traj in stat[str(j)][0]:
        st = []
        for task in traj:
            st.append(TASK_ELEMENTS[task[0][0]])
        traj_B.append(st)

    for A, B in zip(traj_A[:10], traj_B[:10]):
        distance.append(edit_distance(A, B))

    edit_diffs = np.array(distance)
    stat, p_value = wilcoxon(edit_diffs)

    return stat, p_value


def test(file_path):
    with open(file_path + '.json', 'r') as f:
        stat = json.load(f)

    for i in range(0, K - 1):
        for j in range(i + 1, K):
            print(i, j)
            print(wilcoxon_test(stat, i, j))


if __name__ == '__main__':
    # file_path = ('experiments/hrl/kitchen/prior_bc/eval/skill_evaluate_20250509_162122.json')
    # plot_skill(file_path)
    # plot_task_transition(file_path, aggregate=True)
    # test(file_path)

    # file = '/home/wenyongyan/文档/spirl-master/spirl/experiments/hrl/kitchen/spirl_cl_vq/mkbl_s0_k16_inverse_kl/episode_3.h5'
    for i in range(10):
        file = f'experiments/hrl/kitchen/ppo/kbts_s0_hi10/rollout_{i}.h5'
        plot_episode(file)
