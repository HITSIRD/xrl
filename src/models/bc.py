import torch
import torch.nn as nn

from src.components.model import BaseModel
from src.modules.distributions import Categorical
from src.modules.networks import CNNEncoder, ResNetEncoder
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
        if hasattr(self._hp, 'use_resnet') and self._hp.use_resnet:
            return ResNetEncoder(self._hp.img_enc_dim)
        else:
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

        self.prior_head = self._build_prior_head(hp)

    def _build_prior_head(self, hp):
        return nn.Sequential(
            nn.Linear(hp.img_enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hp.skill_dim),
            nn.LogSoftmax(dim=1)
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
            # only prior output
            return self.prior_head(self.prior_encoder(input))

    def loss(self, output, inputs):
        losses = AttrDict()

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()
        # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions)
        losses.prior = nll_loss(output.prior, inputs.skills.argmax(dim=-1))
        # losses.prior = kl_loss(output.prior, self._smooth_one_hot(inputs.skills.argmax(dim=-1),
        #                                                           n_classes=self._hp.skill_dim,
        #                                                           smoothing=0.05))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def compute_learned_prior(self, obs):
        logits = self.prior_head(self.prior_encoder(obs))
        return Categorical(logits=logits)

    def _smooth_one_hot(self, targets, n_classes, smoothing=0.1):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            true_dist = torch.zeros(size=(targets.size(0), n_classes), device=targets.device)
            true_dist.fill_(smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        return true_dist

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
        self._logger.log_scalar(model_output.prior_entropy, "prior_entropy", step, phase)  # wandb
        # self._logger.add_scalar(f'{phase}/prior_entropy', model_output.prior_entropy, step)

        # log videos/gifs in tensorboard
        if log_images:
            print('{} {}: logging videos'.format(phase, step))
            self._logger.visualize(model_output, inputs, losses, step, phase, logger, **logging_kwargs)


class MultiStepsOneHotImagePriorBCModel(OneHotImagePriorBCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        self.prior_head_1 = self._build_prior_head(hp)
        self.prior_head_2 = self._build_prior_head(hp)

    def forward(self, input, future_output=False):
        if isinstance(input, AttrDict):
            output = AttrDict()

            img_embed = self.encoder(input.images)
            output.reconstruction = self.head(torch.cat([img_embed, input.skills], dim=-1))

            enc = self.prior_encoder(input.images)
            output.prior = self.prior_head(enc)
            output.prior_1 = self.prior_head_1(enc)
            output.prior_2 = self.prior_head_2(enc)

            output.prior_probs = torch.exp(output.prior)
            output.prior_entropy = -torch.sum(output.prior_probs * output.prior, dim=1).mean()
            return output
        else:
            # only prior output
            enc = self.prior_encoder(input)
            if future_output:
                return self.prior_head(enc), self.prior_head_1(enc), self.prior_head_2(enc)
            return self.prior_head(enc)

    def loss(self, output, inputs):
        losses = AttrDict()

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()
        # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions)
        losses.prior = nll_loss(output.prior, inputs.skills.argmax(dim=-1))\
                        + nll_loss(output.prior_1, inputs.future_skills[:, 0].argmax(dim=-1))\
                        + nll_loss(output.prior_2, inputs.future_skills[:, 1].argmax(dim=-1))
        # losses.prior = kl_loss(output.prior, self._smooth_one_hot(inputs.skills.argmax(dim=-1),
        #                                                           n_classes=self._hp.skill_dim,
        #                                                           smoothing=0.05))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def compute_learned_prior(self, obs):
        enc = self.prior_encoder(obs)
        logits_0 = self.prior_head(enc)
        logits_1 = self.prior_head_1(enc)
        logits_2 = self.prior_head_2(enc)
        return Categorical(logits=logits_0), Categorical(logits=logits_1), Categorical(logits=logits_2)


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
        self._logger.log_scalar(model_output.prior_entropy, 'prior_entropy', step, phase)
        # self._logger.log_scalar(model_output.precision, 'precision', step, phase)
        # self._logger.log_scalar(model_output.recall, 'recall', step, phase)
        # self._logger.log_scalar(model_output.f1_score, 'f1_score', step, phase)

        # self._logger.add_scalar(f'{phase}/prior_entropy', model_output.prior_entropy, step)
        # self._logger.add_scalar(f{phase}/precision', model_output.precision, step)
        # self._logger.add_scalar(f'{phase}/recall', model_output.recall, step)
        # self._logger.add_scalar(f'{phase}/f1_score', model_output.f1_score, step)
