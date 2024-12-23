import os

from skilltree.models.closed_loop_spirl_mdl import ClSPiRLMdl, ImageClSPiRLMdl
from skilltree.components.logger import Logger
from skilltree.utils.general_utils import AttrDict
from skilltree.configs.default_data_configs.kitchen import data_spec
from skilltree.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageClSPiRLMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'kitchen/kitchen-mixed-v0'),
    'epoch_cycles_train': 50,
    'num_epochs': 100,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    # state_dim=data_spec.state_dim,
    image_res=data_spec.res,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    n_input_frames=1,
    cond_decode=True,
    prior_input_res=200,

)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames  # flat last action from seq gets cropped
