import os

from strl.components.logger import Logger
from strl.models.skill_prior_mdl import SkillSpaceLogger
from strl.models.closed_loop_spirl_mdl import ClSPiRLMdl
from strl.utils.general_utils import AttrDict
from strl.configs.default_data_configs.maze import data_spec
from strl.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ClSPiRLMdl,
    # 'logger': SkillSpaceLogger,
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
    prior_input_res=data_spec.res,
    n_processing_layers=5,
    # n_input_frames=2,
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
# data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames  # flat last action from seq gets cropped
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
