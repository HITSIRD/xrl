import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

K = 16
# plt.rcParams['text.usetex'] = True

def get_filenames_recursive(directory):
    filenames = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            filenames.append(path)
        elif os.path.isdir(path):
            filenames.extend(get_filenames_recursive(path))
    return filenames

def plot_skill(file_path):
    dataset = h5py.File(file_path, 'r')

    skill = np.zeros((len(dataset['traj']['hl_action_index']), K))
    for i, s in enumerate(dataset['traj']['hl_action_index']):
        skill[i, s] = 1

    plt.figure(figsize=(4, 1.8))
    label_size = 5

    x_labels = [i for i in range(len(dataset['traj']['hl_action_index']))]
    y_labels = [i for i in range(K)]

    plt.yticks(np.arange(K), size=label_size)
    plt.xticks(np.arange(len(x_labels)), labels=x_labels, size=label_size)
    # plt.title("Skill Evaluation", size=label_size)

    # for i in range(K):
    #     for j in range(len(TASK_ELEMENTS)):
    #         plt.text(i, j, rate[i, j], ha="center", va="center", color="w", size=label_size)

    plt.xlabel('Time Step (× 10)', size=8)
    plt.ylabel('Skill Index', size=8)
    plt.imshow(skill.T, aspect='auto', cmap='viridis')
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=label_size)
    plt.tight_layout()
    # plt.title('')
    # plt.show()
    plt.savefig(file_path + '.pdf')


if __name__ == '__main__':
    plot_skill(
        '/home/wenyongyan/文档/spirl-master/spirl/experiments/hrl/kitchen/spirl_cl_vq/mkbl_s0_k16_inverse_kl/episode_3.h5')
