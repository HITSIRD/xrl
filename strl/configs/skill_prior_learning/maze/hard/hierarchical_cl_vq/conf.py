import os

from strl.models.closed_loop_vq_spirl_mdl import ClVQSPiRLMdl
from strl.components.logger import Logger
from strl.utils.general_utils import AttrDict
from strl.configs.default_data_configs.maze import data_spec
from strl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ClVQSPiRLMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'maze/maze2d_40_seed0'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-3,
    n_processing_layers=5,
    cond_decode=True,
    codebook_K=16,
    commitment_beta=0.25,
    fixed_codebook=False,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
