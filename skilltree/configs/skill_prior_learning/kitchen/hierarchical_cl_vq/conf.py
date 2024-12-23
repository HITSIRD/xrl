import os

from skilltree.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl, ImageClVQSPiRLMdl
from skilltree.components.logger import Logger
from skilltree.utils.general_utils import AttrDict
from skilltree.configs.default_data_configs.kitchen import data_spec
from skilltree.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageClVQSPiRLMdl,
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
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
    n_input_frames=1,
    prior_input_res=200,
    codebook_K=16,
    commitment_beta=0.25,
    fixed_codebook=False,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames  # flat last action from seq gets cropped
