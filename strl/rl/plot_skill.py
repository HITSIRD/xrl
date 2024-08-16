import matplotlib.pyplot as plt
import numpy as np
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

TASK_ELEMENTS = {'microwave': 0,
                 'kettle': 1,
                 'bottom burner': 2,
                 'top burner': 3,
                 'light switch': 4,
                 'slide cabinet': 5,
                 'hinge cabinet': 6
                 }

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

K = 16


def plot_skill(file_path):
    with open(file_path + '.json', 'r') as f:
        stat = json.load(f)

    rate = np.zeros((K, len(TASK_ELEMENTS)))
    for z in stat.keys():
        for k, v in stat[z][1].items():
            rate[int(z)][TASK_ELEMENTS[k]] = v

    plt.figure(figsize=(4.3, 1.6))
    label_size = 6

    x_labels = [i for i in range(K)]
    y_labels = TASK_ELEMENTS.keys()

    plt.xticks(np.arange(K), labels=x_labels, rotation_mode="anchor", ha="right", size=label_size)
    plt.yticks(np.arange(len(TASK_ELEMENTS)), labels=y_labels, size=label_size)
    # plt.title("Skill Evaluation", size=label_size)

    # plt.axis.xaxis.

    # for i in range(K):
    #     for j in range(len(TASK_ELEMENTS)):
    #         plt.text(i, j, rate[i, j], ha="center", va="center", color="w", size=label_size)

    plt.imshow(rate.T)
    cb = plt.colorbar(fraction=0.0204, pad=0.05)
    cb.ax.tick_params(labelsize=label_size)
    plt.tight_layout(pad=0.2)
    plt.savefig(file_path + '.pdf')


def plot_task_transition(file_path, aggregate=False):
    with open(file_path + '.json', 'r') as f:
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


if __name__ == '__main__':
    # file_path = 'hrl/kitchen/oracle_vq/mkbl/skill_evaluate_prior_mkbl_3'
    # file_path = 'hrl/kitchen/oracle_vq/mkbl/new_reconstruction_loss_1'
    # file_path = 'hrl/kitchen/oracle_vq/mlsh_finetune/skill_evaluate_prior_mlsh_3'
    # file_path = 'hrl/calvin/oracle_vq/K_16/skill_evaluate_prior_2'
    # file_path = 'hrl/calvin/oracle_vq/finetune/skill_evaluate_prior_0'
    # file_path = 'hrl/kitchen/oracle_vq/mkbl/new_reconstruction'
    file_path = '/strl/experiments/hrl/kitchen/vq/mkbl/skill_evaluation'
    # plot_skill(file_path)
    plot_task_transition(file_path, aggregate=True)
