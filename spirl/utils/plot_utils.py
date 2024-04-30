# 绘制代码
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import copy
import matplotlib as mpl

def get_binary_index(tree):
    """
    Get binary index for tree nodes:
    From

    0
    1 2
    3 4 5 6 

    to 

    '0'
    '00' '01' 
    '000' '001' '010' '011'

    """
    index_list = []
    for layer_idx in range(0, tree.max_depth+1):
        index_list.append([bin(i)[2:].zfill(layer_idx+1) for i in range(0, np.power(2, layer_idx))])
    return np.concatenate(index_list)

def path_from_prediction(tree, idx):
    """
    Generate list of nodes as decision path, 
    with each node represented by a binary string and an int index
    """
    binary_idx_list = []
    int_idx_list=[]
    idx = int(idx)
    for layer_idx in range(tree.max_depth+1, 0, -1):
        binary_idx_list.append(bin(idx)[2:].zfill(layer_idx))
        int_idx_list.append(2**(layer_idx-1)-1+idx)
        idx = int(idx/2)
    binary_idx_list.reverse()  # from top to bottom
    int_idx_list.reverse() 
    return binary_idx_list, int_idx_list

def draw_tree(original_tree, input_img=None, show_correlation=False, DrawTree=None, savepath=''):
    '''
    Need to carefully select several configurations for well displaying trees for different environments, e.g. CartPole and LunarLander-v2
    '''

    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import ConnectionPatch

    tree = copy.copy(original_tree)
    if DrawTree=='FL': # draw the feature learning tree
        tree.inner_node_num = tree.num_fl_inner_nodes
        tree.max_depth = tree.feature_learning_depth
        tree.leaf_num = tree.num_fl_leaves
        inner_nodes_name='fl_inner_nodes.weight'
        leaf_nodes_name='fl_leaf_weights'
        input_shape=(tree.input_dim,)

    elif DrawTree == 'DM':  # draw the decision making tree
        tree.inner_node_num = tree.num_dc_inner_nodes
        tree.max_depth = tree.decision_depth
        tree.leaf_num = tree.num_dc_leaves
        inner_nodes_name='dc_inner_nodes.weight'
        leaf_nodes_name='dc_leaves'
        input_shape=(tree.num_intermediate_variables,)
        # input_img=tree.max_feature_value.squeeze().detach().cpu().numpy()  # replace the original input image to be intermediate feature value

    def _add_arrow(ax_parent, ax_child, xyA, xyB, color='black', linestyle=None):
        '''Private utility function for drawing arrows between two axes.'''
        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data',
                              axesA=ax_child, axesB=ax_parent, arrowstyle='<|-',
                              color=color, linewidth=tree.max_depth, linestyle=linestyle)
        ax_child.add_artist(con)

    inner_nodes = tree.state_dict()[inner_nodes_name]
    leaf_nodes = tree.state_dict()[leaf_nodes_name]
    binary_indices = get_binary_index(tree)
    inner_indices = binary_indices[:tree.inner_node_num]
    leaf_indices = binary_indices[tree.inner_node_num:]
    
    if len(input_shape) == 3:
        img_rows, img_cols, img_chans = input_shape
    elif len(input_shape) == 1:
        img_rows, img_cols = input_shape[0], input_shape[0]

    if DrawTree == 'FL':  # each leaf contains vectors of number: tree.args['num_intermediate_variables'] 
        leaf_nodes = leaf_nodes.view(tree.leaf_num, tree.num_intermediate_variables, tree.input_dim)

    kernels = dict([(node_idx, node_value.cpu().numpy().reshape(input_shape)) for node_idx, node_value in zip (inner_indices, inner_nodes[:, 1:]) ])
    biases = dict([(node_idx, node_value.cpu().numpy().squeeze()) for node_idx, node_value in zip (inner_indices, inner_nodes[:, :1]) ])
    leaves = dict([(leaf_idx, np.array([leaf_dist.cpu().numpy()])) for leaf_idx, leaf_dist in zip (leaf_indices, leaf_nodes) ])
    n_leaves = tree.leaf_num
    assert len(leaves) == n_leaves

    # prepare figure and specify grid for subplots
    # fig = plt.figure(figsize=(n_leaves, n_leaves//2), facecolor=(0.5, 0.5, 0.0, 0.8))
    # fig = plt.figure(figsize=(n_leaves, n_leaves//2), facecolor='grey')  # for lunarlander
    # fig = plt.figure(figsize=(2*n_leaves, n_leaves), facecolor='grey')  # for cartpole
    fig = plt.figure(figsize=(2*n_leaves, n_leaves), facecolor='grey')  # for cartpole


    gs = GridSpec(tree.max_depth+1, n_leaves*2,
                  height_ratios=[1]*tree.max_depth+[0.5])

    # Grid Coordinate X (horizontal)
    gcx = [list(np.arange(1, 2**(i+1), 2) * (2**(tree.max_depth+1) // 2**(i+1)))
           for i in range(tree.max_depth+1)]
    gcx = list(itertools.chain.from_iterable(gcx))
    axes = {}
    path = ['0']

    imshow_args = {'origin': 'upper', 'interpolation': 'None', 'cmap': plt.get_cmap('coolwarm')}
    # get mininal and maximal values for kernels and leaves separately, to give proper color mapping ranges.
    # kernel_min_val = np.min(list(kernels.values()))
    # kernel_max_val  = np.max(list(kernels.values()))
    # leaf_min_val = np.min(list(leaves.values()))
    # leaf_max_val  = np.max(list(leaves.values()))
    kernel_min_val = -10000.  # for lunarlander
    kernel_max_val = 10000.   # for lunarlander
    # kernel_min_val = -50.  # for cartpole DM
    # kernel_max_val = 50.  # for cartpole DM
    # kernel_min_val = -5.  # for cartpole FL
    # kernel_max_val = 5.  # for cartpole FL
    leaf_min_val = 0.
    leaf_max_val = 1.
        
    # draw tree nodes
    for pos, key in enumerate(sorted(kernels.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1, gcx[pos]-2:gcx[pos]+2])
        axes[key] = ax
        # kernel_image = np.abs(kernels[key])  # absolute value
        # kernel_image = kernel_image/np.sum(kernel_image)  # normalization
        kernel_image = kernels[key]

        if len(kernel_image.shape)==3: # 2D image (H, W, C)
            ax.imshow(kernel_image.squeeze(), vmin=kernel_min_val, vmax=kernel_max_val, **imshow_args)
        elif len(kernel_image.shape)==1:
            vector_image = np.ones((kernel_image.shape[0], 1)) @ [kernel_image]
            ax.imshow(vector_image, vmin=kernel_min_val, vmax=kernel_max_val, **imshow_args)
        ax.axis('off')
        if DrawTree!='FL':  # feature learning tree do not have titile indicating the classification 
            digits = set([np.argmax(leaves[k]) for k in leaves.keys()
                        if k.startswith(key)])
            title = ','.join(str(digit) for digit in digits)
            plt.title('{}'.format(title))
                
    # # change the way to get path to be via the prediction by the tree
    # if DrawTree=='FL':
    #     max_leaf_idx = tree.max_leaf_idx_fl
    # elif DrawTree=='DM':
    #     max_leaf_idx = tree.max_leaf_idx_dc
    # path, _ = path_from_prediction(tree, max_leaf_idx)

    # draw tree leaves
    for pos, key in enumerate(sorted(leaves.keys(), key=lambda x:(len(x), x))):
        ax = plt.subplot(gs[len(key)-1,
                            gcx[len(kernels)+pos]-1:gcx[len(kernels)+pos]+1])
        axes[key] = ax
        if len(leaves[key].shape)>2:  # output multi-dimension, e.g. intermediate features for feature learning tree
            leaf_image = leaves[key].squeeze(0)
        else:
            leaf_image = np.ones((tree.output_dim, 1)) @ leaves[key]

        ax.imshow(leaf_image, vmin=leaf_min_val, vmax=leaf_max_val, **imshow_args)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if DrawTree!='FL':  # feature learning tree do not have titile indicating the classification 
            plt.title('{}'.format(np.argmax(leaves[key])), y=-.5)
        
    # add arrows indicating flow
    for pos, key in enumerate(sorted(axes.keys(), key=lambda x:(len(x), x))):
        children_keys = [k for k in axes.keys()
                         if len(k) == len(key) + 1 and k.startswith(key)]
        for child_key in children_keys:
            p_rows, p_cols = axes[key].get_images()[0].get_array().shape
            c_rows, c_cols = axes[child_key].get_images()[0].get_array().shape

            # # distinguish with green and red color
            # color = 'green' if (key in path and child_key in path) else 'red'
            # _add_arrow(axes[key], axes[child_key],
            #            (c_cols//2, 1), (p_cols//2, p_rows-1), color)

            # distinguish with solid or dotted lines
            linestyle = None
            _add_arrow(axes[key], axes[child_key],
                       (c_cols//2, 1), (p_cols//2, p_rows-1), color='black', linestyle=linestyle)


    # draw input image with arrow indicating flow into the root node
    if input_img is not None:
        ax = plt.subplot(gs[0, 0:4])  # for lunarlander
        # ax = plt.subplot(gs[0, 0:2])  # for cartpole
        img_min_val = np.min(input_img)
        img_max_val = np.max(input_img)
        if len(input_img.shape)==3: # 2D image (H, W, C)
            ax.imshow(input_img.squeeze(), clim=(0.0, 1.0), vmin=img_min_val, vmax=img_max_val, **imshow_args)
        elif len(input_img.shape)==1:
            vector_image = np.ones((input_img.shape[0], 1)) @ [input_img]
            ax.imshow(vector_image, vmin=img_min_val, vmax=img_max_val, **imshow_args)
        ax.axis('off')
        plt.title('input')
        # # distinguish with green and red color
        # _add_arrow(ax, axes['0'],
        #            (1, img_rows//2), (img_cols-1, img_rows//2), 'green')
        
        # distinguish with solid or dotted lines
        # _add_arrow(ax, axes['0'], (1, img_rows//2), (img_cols-1, img_rows//2), 'black', None)

        norm = mpl.colors.Normalize(vmin=img_min_val,vmax=img_max_val)
        sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
        sm.set_array([])
        cbaxes = fig.add_axes([0.01, 0.7, 0.03, 0.2])  # This is the position for the colorbar
        plt.colorbar(sm, ticks=np.linspace(img_min_val,img_max_val,5), cax = cbaxes)

    # plot color bar for kernels and leaves separately
    norm = mpl.colors.Normalize(vmin=kernel_min_val,vmax=kernel_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.4, 0.03, 0.2])  # This is the position for the colorbar
    plt.colorbar(sm, ticks=np.linspace(kernel_min_val,kernel_max_val,5), cax = cbaxes)

    norm = mpl.colors.Normalize(vmin=leaf_min_val,vmax=leaf_max_val)
    sm = plt.cm.ScalarMappable(cmap=imshow_args['cmap'], norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.01, 0.1, 0.03, 0.2])  # This is the position for the colorbar, second dim is y, from bottom to top in img: 0->1
    plt.colorbar(sm, ticks=np.linspace(leaf_min_val,leaf_max_val,5), cax = cbaxes)


    if savepath:
        plt.savefig(savepath, facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()

def get_path(tree, input, Probs=False):
    tree.forward(torch.Tensor(input).unsqueeze(0))
    max_leaf_idx = tree.max_leaf_idx
    _, path_idx_int = path_from_prediction(tree, max_leaf_idx)
    if Probs:
        return path_idx_int, tree.inner_probs.squeeze().detach().cpu().numpy()
    else:
        return path_idx_int

import torch
from spirl.modules.subnetworks import VQCDTPredictor

# 模型配置
codebook = 16
class HyperParameters:
    def __init__(self):
        self.feature_learning_depth = 0
        self.decision_depth = 5
        self.num_intermediate_variables = 20
        self.greatest_path_probability = False
        self.beta_fl = False
        self.beta_dc = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hp = HyperParameters()

tree = VQCDTPredictor(hp, 60, codebook)
tree.load_model('/home/zuo/project/xrl/spirl/experiments/cdt_model/16+0+5+20.pth')

# 参数打印
num_params = 0
for key, v in tree.state_dict().items():
    print(key, v.reshape(-1).shape[0])
    num_params+=v.reshape(-1).shape[0]
print('Total number of parameters in model: ', num_params)

draw_tree(tree, input_img=None, DrawTree='DM', savepath='/home/zuo/project/xrl/spirl/experiments/cdt_model/16+0+5+20+test.png')

# draw_tree_with_params(tree, savepath='/home/zuo/project/xrl/spirl/experiments/cdt_model/16+0+5+20+test.png')
