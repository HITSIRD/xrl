import matplotlib.pyplot as plt
import numpy as np
import json

TASK_ELEMENTS = {'microwave': 0,
                 'kettle': 1,
                 'bottom burner': 2,
                 'top burner': 3,
                 'light switch': 4,
                 'slide cabinet': 5,
                 'hinge cabinet': 6
                 }
K = 16


def plot_skill(file_path):
    with open(file_path + '.json', 'r') as f:
        stat = json.load(f)

    rate = np.zeros((K, len(TASK_ELEMENTS)))
    for z in stat.keys():
        for k, v in stat[z][1].items():
            rate[int(z)][TASK_ELEMENTS[k]] = v

    plt.figure(figsize=(7, 2))
    label_size = 6

    x_labels = [i for i in range(K)]
    y_labels = TASK_ELEMENTS.keys()

    plt.xticks(np.arange(K), labels=x_labels, rotation_mode="anchor", ha="right", size=label_size)
    plt.yticks(np.arange(len(TASK_ELEMENTS)), labels=y_labels, size=label_size)
    plt.title("Skill Evaluation", size=label_size)

    # plt.axis.xaxis.

    # for i in range(K):
    #     for j in range(len(TASK_ELEMENTS)):
    #         plt.text(i, j, rate[i, j], ha="center", va="center", color="w", size=label_size)

    plt.imshow(rate.T)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=label_size)
    plt.tight_layout()
    plt.savefig(file_path + '.pdf')


if __name__ == '__main__':
    # file_path = 'hrl/kitchen/vq/test/refined'
    file_path = 'hrl/kitchen/vq/test/skill_evaluate_20240419_103610'
    plot_skill(file_path)
