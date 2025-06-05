import torch
import torch.nn as nn

from src.components.model import BaseModel
from src.modules.distributions import Categorical
from src.modules.networks import CNNEncoder
from src.utils.general import AttrDict
import torch.nn.functional as F


class BCModel(BaseModel):
    def __init__(self, hp, logger):
        super().__init__(logger)
        self._hp = hp
        self.device = self._hp.device

        self.encoder = self.build_encoder()
        self.policy_head = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, hp.action_dim)
        )

    def build_encoder(self):
        pass

    def forward(self, input):
        pass

    @property
    def latent_dim(self):
        return self._hp.skill_dim


class OneHotImageBCModel(BCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        self.head = nn.Sequential(
            nn.Linear(hp.img_enc_dim + hp.skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hp.action_dim)
        )

    def build_encoder(self):
        return CNNEncoder(3, self._hp.prior_input_res, self._hp.img_enc_dim)

    def forward(self, input):
        output = AttrDict()

        img_embed = self.encoder(input.images)
        output.reconstruction = self.head(torch.cat([img_embed, input.skills], dim=-1))
        return output

    def loss(self, output, inputs):
        losses = AttrDict()

        mse_loss = torch.nn.MSELoss()

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions)

        losses.total = losses.rec_mse
        return losses

    def unflatten_obs(self, raw_obs):
        """Utility to unflatten [obs, prior_obs] concatenated observation (for RL usage)."""
        assert raw_obs.shape[1] == self._hp.state_dim \
               + self._hp.prior_input_res ** 2 * 3 * self._hp.n_input_frames
        return AttrDict(
            obs=raw_obs[:, :self._hp.state_dim],
            prior_obs=raw_obs[:, self._hp.state_dim:].reshape(raw_obs.shape[0], 3 * self._hp.n_input_frames,
                                                              self._hp.prior_input_res, self._hp.prior_input_res)
        )


class OneHotImagePriorBCModel(OneHotImageBCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        self.prior_encoder = self.build_encoder()

        self.prior_head = nn.Sequential(
            nn.Linear(hp.img_enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hp.skill_dim),
            nn.LogSoftmax()
        )

    def forward(self, input):
        if isinstance(input, AttrDict):
            output = AttrDict()

            img_embed = self.encoder(input.images)
            output.reconstruction = self.head(torch.cat([img_embed, input.skills], dim=-1))
            output.prior = self.prior_head(self.prior_encoder(input.images))

            output.prior_probs = torch.exp(output.prior)
            output.prior_entropy = -torch.sum(output.prior_probs * output.prior, dim=1).mean()
            return output
        else:
            return self.prior_head(self.prior_encoder(input))

    def loss(self, output, inputs):
        losses = AttrDict()

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions)
        losses.prior = nll_loss(output.prior, inputs.skills.argmax(dim=-1))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def compute_learned_prior(self, obs):
        logits = self.prior_head(self.prior_encoder(obs))
        return Categorical(logits=logits)

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        """Optionally visualizes outputs of SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of SPIRL model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        # self._logger.log_scalar(model_output.prior_entropy, "prior_entropy", step, phase) # wandb
        self._logger.add_scalar(f'{phase}/prior_entropy', model_output.prior_entropy, step)

        # log videos/gifs in tensorboard
        if log_images:
            print('{} {}: logging videos'.format(phase, step))
            self._logger.visualize(model_output, inputs, losses, step, phase, logger, **logging_kwargs)


class OneHotImagePriorCompleteBCModel(OneHotImagePriorBCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        self.complete_encoder = self.build_encoder()

        self.classifier = nn.Sequential(
            nn.Linear(hp.img_enc_dim + hp.skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input):
        output = AttrDict()

        img_embed = self.encoder(input.images)
        output.reconstruction = self.head(torch.cat([img_embed, input.skills], dim=-1))
        output.prior = self.prior_head(self.prior_encoder(input.images))

        prior_probs = torch.exp(output.prior)
        output.prior_entropy = -torch.sum(prior_probs * output.prior, dim=1).mean()

        c_img_embed = self.complete_encoder(input.images)
        output.complete = self.classifier(torch.cat([c_img_embed, input.skills], dim=-1))

        # if hasattr(input, 'complete'):
        #     preds = output.complete.view(-1)
        #     targets = input.complete.view(-1)
        #
        #     preds = torch.sigmoid(preds)
        #     pred_labels = (preds > 0.5).float()
        #
        #     TP = ((pred_labels == 1) & (targets == 1)).sum().item()
        #     FP = ((pred_labels == 1) & (targets == 0)).sum().item()
        #     FN = ((pred_labels == 0) & (targets == 1)).sum().item()
        #
        #     output.precision = TP / (TP + FP + 1e-8)
        #     output.recall = TP / (TP + FN + 1e-8)
        #     output.f1_score = 2 * output.precision * output.recall / (output.precision + output.recall + 1e-8)
        return output

    def _focal_loss(self, inputs, targets, alpha=0.25, gamma=2):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        F_loss = alpha * (1 - pt) ** gamma * bce_loss
        return F_loss.mean()

    def loss(self, output, inputs):
        losses = AttrDict()

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()
        # bce_loss = torch.nn.BCEWithLogitsLoss()

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions)
        losses.prior = nll_loss(output.prior, inputs.skills.argmax(dim=-1))
        losses.complete = self._focal_loss(output.complete, inputs.complete.unsqueeze(-1).float())

        losses.total = losses.rec_mse + losses.prior + losses.complete
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        """Optionally visualizes outputs of SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg losses: output of SPIRL model loss() function
        :arg step: current training iteration
        :arg log_images: if True, log image visualizations (otherwise only scalar losses etc get logged automatically)
        :arg phase: 'train' or 'val'
        :arg logger: logger class, visualization functions should be implemented in this class
        """
        # self._logger.log_scalar(model_output.prior_entropy, 'prior_entropy', step, phase)
        # self._logger.log_scalar(model_output.precision, 'precision', step, phase)
        # self._logger.log_scalar(model_output.recall, 'recall', step, phase)
        # self._logger.log_scalar(model_output.f1_score, 'f1_score', step, phase)

        self._logger.add_scalar(f'{phase}/prior_entropy', model_output.prior_entropy, step)
        # self._logger.add_scalar(f{phase}/precision', model_output.precision, step)
        # self._logger.add_scalar(f'{phase}/recall', model_output.recall, step)
        # self._logger.add_scalar(f'{phase}/f1_score', model_output.f1_score, step)
