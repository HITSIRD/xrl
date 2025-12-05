import torch
import torch.nn as nn

from src.components.model import BaseModel
from src.modules.distributions import Categorical
from src.modules.networks import CNNEncoder, ResNetEncoder
from src.utils.general import AttrDict
import torch.nn.functional as F
from contextlib import contextmanager


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
            output.prior_entropy = -torch.sum(output.prior_probs * output.prior, dim=-1).mean()
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


class SequenceOneHotImagePriorBCModel(OneHotImagePriorBCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=hp.img_enc_dim,
            hidden_size=hp.rnn_hidden_dim,
            num_layers=hp.rnn_num_layers,
            batch_first=True
        )

        self._image = None
        self._hidden_state = None
        self._analyze_hidden_mode = False

    def _build_prior_head(self, hp):
        return nn.Sequential(
            nn.Linear(hp.rnn_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hp.skill_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input, update_hidden=False, hidden_state=None):
        """
        inputs.images: [B, T, C, H, W]
        inputs.skills: [B, T, skill_dim]
        inputs.pad_mask: [B, T]
        """
        if self._analyze_hidden_mode:
            self._image.requires_grad_(True)
            img_feature = self.prior_encoder(self._image)
            if input.shape[0] != 1:
                input = input.unsqueeze(0)
                img_feature = img_feature.expand((input.shape[-2], *img_feature.shape))
            gru_output, _ = self.gru(img_feature, input)
            gru_output = gru_output.squeeze(1)
            out = self.prior_head(gru_output)
            return out

        if isinstance(input, AttrDict):
            output = AttrDict()

            B, T = input.images.shape[:2]

            img_embed = self.encoder(input.images.view(B * T, *input.images.shape[2:]))
            output.reconstruction = self.head(
                torch.cat([img_embed, input.skills.view(B * T, *input.skills.shape[2:])], dim=-1))

            prior_img_embed = self.prior_encoder(input.images.view(B * T, *input.images.shape[2:]))
            prior_img_embed, _ = self.gru(prior_img_embed.view(B, T, -1))
            output.prior = self.prior_head(prior_img_embed)

            output.prior_probs = torch.exp(output.prior)
            output.prior_entropy = -torch.sum(output.prior_probs * output.prior, dim=-1).mean()
            return output
        else:
            # only prior output
            return self._forward_single_step(input, update_hidden=update_hidden, hidden_state=hidden_state)

    def _forward_single_step(self, image, update_hidden=False, hidden_state=None):
        img_feature = self.prior_encoder(image)
        h = self._hidden_state if hidden_state is None else hidden_state
        gru_output, hidden_state = self.gru(img_feature, h)
        if update_hidden:
            self._hidden_state = hidden_state
        return self.prior_head(gru_output)

    def reset_hidden_state(self):
        self._hidden_state = None

    @contextmanager
    def enable_hidden_analysis(self):
        self._analyze_hidden_mode = True
        yield ()
        self._analyze_hidden_mode = False

    def loss(self, output, inputs):
        losses = AttrDict()
        B, T, C = output.prior.shape

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions.view(B * T, *inputs.actions.shape[2:]))
        losses.prior = nll_loss(output.prior.reshape(B * T, C), inputs.skills.argmax(dim=-1).reshape(B * T))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def compute_learned_prior(self, obs):
        return Categorical(logits=self._forward_single_step(obs))

    def set_image(self, image):
        self._image = image

    def hidden_state(self):
        return self._hidden_state


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

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions)
        losses.prior = nll_loss(output.prior, inputs.skills.argmax(dim=-1)) \
                       + nll_loss(output.prior_1, inputs.future_skills[:, 0].argmax(dim=-1)) \
                       + nll_loss(output.prior_2, inputs.future_skills[:, 1].argmax(dim=-1))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def compute_learned_prior(self, obs):
        enc = self.prior_encoder(obs)
        logits_0 = self.prior_head(enc)
        logits_1 = self.prior_head_1(enc)
        logits_2 = self.prior_head_2(enc)
        return Categorical(logits=logits_0), Categorical(logits=logits_1), Categorical(logits=logits_2)


class MultiStepsSequenceOneHotImagePriorBCModel(SequenceOneHotImagePriorBCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        self.prior_head_1 = self._build_prior_head(hp)
        self.prior_head_2 = self._build_prior_head(hp)

    def forward(self, input, future_output=False, update_hidden=False):
        """
        inputs.images: [B, T, C, H, W]
        inputs.skills: [B, T, skill_dim]
        inputs.pad_mask: [B, T]
        """
        if isinstance(input, AttrDict):
            output = AttrDict()

            B, T = input.images.shape[:2]

            img_embed = self.encoder(input.images.view(B * T, *input.images.shape[2:]))
            output.reconstruction = self.head(
                torch.cat([img_embed, input.skills.view(B * T, *input.skills.shape[2:])], dim=-1))

            prior_img_embed = self.prior_encoder(input.images.view(B * T, *input.images.shape[2:]))
            prior_img_embed, _ = self.gru(prior_img_embed.view(B, T, -1))
            output.prior = self.prior_head(prior_img_embed)
            output.prior_1 = self.prior_head_1(prior_img_embed)
            output.prior_2 = self.prior_head_2(prior_img_embed)

            output.prior_probs = torch.exp(output.prior)
            output.prior_entropy = -torch.sum(output.prior_probs * output.prior, dim=-1).mean()
            return output
        else:
            # only prior output
            return self._forward_single_step(input, future_output, update_hidden)

    def _forward_single_step(self, image, future_output, update_hidden=False):
        img_feature = self.prior_encoder(image)
        gru_output, hidden_state = self.gru(img_feature, self.hidden_state)
        if update_hidden:
            self.hidden_state = hidden_state
        if future_output:
            return self.prior_head(gru_output), self.prior_head_1(gru_output), self.prior_head_2(gru_output)
        else:
            return self.prior_head(gru_output)

    def reset_hidden_state(self):
        print('reset hidden state...')
        self.hidden_state = None

    def loss(self, output, inputs):
        losses = AttrDict()
        B, T, C = output.prior.shape

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions.view(B * T, *inputs.actions.shape[2:]))
        losses.prior = nll_loss(output.prior.reshape(B * T, C), inputs.skills.argmax(dim=-1).reshape(B * T)) + nll_loss(
            output.prior_1.reshape(B * T, C), inputs.future_skills[:, 0].argmax(dim=-1).reshape(B * T)) + nll_loss(
            output.prior_2.reshape(B * T, C), inputs.future_skills[:, 1].argmax(dim=-1).reshape(B * T))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def compute_learned_prior(self, obs):
        logits_0, logits_1, logits_2 = self._forward_single_step(obs, future_output=True, update_hidden=True)
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


class TransformerOneHotImagePriorBCModel(OneHotImagePriorBCModel):
    def __init__(self, hp, logger):
        super().__init__(hp, logger)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hp.transformer_hidden_dim,
            nhead=hp.transformer_num_head,
            batch_first=True  # requires PyTorch with batch_first support
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=hp.transformer_num_layers)

        # keep prior_head unchanged (expects rnn_hidden_dim in)
        self._image = None
        # replace hidden state with memory buffer for single-step forward
        self._memory = None  # will hold tensor of shape (B, L, d_model) or None
        self._analyze_hidden_mode = False

    def _build_prior_head(self, hp):
        return nn.Sequential(
            nn.Linear(hp.transformer_hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hp.skill_dim),
            nn.LogSoftmax(dim=-1)
        )


    def _transformer_encode(self, seq_feats, pad_mask=None):
        """
        seq_feats: (B, T, img_enc_dim) OR already projected (B, T, d_model)
        pad_mask: (B, T) float mask where 1 indicates valid token (same as original pad_mask)
        returns: encoded (B, T, d_model)
        """

        # transformer expects src_key_padding_mask with True for positions that should be masked
        if pad_mask is not None:
            # pad_mask: 1 for valid, 0 for padded (per original code)
            src_key_padding_mask = (pad_mask == 0)  # shape (B, T) bool
        else:
            src_key_padding_mask = None

        # pass through transformer
        encoded = self.transformer(seq_feats, src_key_padding_mask=src_key_padding_mask)
        return encoded  # (B, T, d_model)

    def forward(self, input, update_hidden=False):
        """
        inputs.images: [B, T, C, H, W]
        inputs.skills: [B, T, skill_dim]
        inputs.pad_mask: [B, T]
        """
        if isinstance(input, AttrDict):
            output = AttrDict()
            B, T = input.images.shape[:2]

            # 1) reconstruction head (unchanged)
            img_embed = self.encoder(input.images.view(B * T, *input.images.shape[2:]))
            output.reconstruction = self.head(
                torch.cat([img_embed, input.skills.view(B * T, *input.skills.shape[2:])], dim=-1))

            # 2) prior: encode each image with prior_encoder
            prior_img_embed = self.prior_encoder(input.images.view(B * T, *input.images.shape[2:]))
            D = prior_img_embed.shape[-1]
            prior_img_embed = prior_img_embed.view(B, T, D)  # (B, T, D)

            pad_mask = input.pad_mask if hasattr(input, 'pad_mask') else None

            # Prepare container for per-time-step combined features
            combined_feats = []
            for t in range(T):
                # history: frames [0:t)
                if t == 0:
                    # No history: use zero vector as history encoding (same device / dtype)
                    history_encoded = prior_img_embed.new_zeros((B, D))
                else:
                    # Encode history tokens with transformer encoder.
                    # _transformer_encode expects (B, L, D) and pad_mask for that window.
                    hist_tokens = prior_img_embed[:, :t, :]  # (B, t, D)
                    if pad_mask is not None:
                        hist_mask = pad_mask[:, :t]  # (B, t)
                    else:
                        hist_mask = None
                    # _transformer_encode should return (B, L, D). We pool (take last non-padded token or mean).
                    hist_encoded = self._transformer_encode(hist_tokens, pad_mask=hist_mask)  # (B, t, D)
                    history_encoded = hist_encoded[:, -1, :]  # (B, D)

                # current frame embedding (not passed through transformer history)
                current_embed = prior_img_embed[:, t, :]  # (B, D)

                # combine history + current. For minimal change, use elementwise add.
                # If you prefer concatenation, adjust prior_head input dims accordingly.
                combined = torch.cat([history_encoded, current_embed], -1)  # (B, D)
                combined_feats.append(combined)

            combined_feats = torch.stack(combined_feats, dim=1)
            output.prior = self.prior_head(combined_feats)

            output.prior_probs = torch.exp(output.prior)
            output.prior_entropy = -torch.sum(output.prior_probs * output.prior, dim=-1).mean()
            return output

        else:
            return self._forward_single_step(input, update_hidden=update_hidden)

    def _forward_single_step(self, image, update_hidden=False):
        """
        image: (B, C, H, W) or (C,H,W) for single
        hidden_state: optional previous memory tensor of shape (B, L, d_model)
        We will maintain a simple per-batch memory buffer self._memory to accumulate last L features.
        """
        # get current image feature
        img_feature = self.prior_encoder(image)  # shape (B, img_enc_dim) or (img_enc_dim)
        memory = self._memory

        if memory is None:
            seq = torch.zeros_like(img_feature)
            new_memory = img_feature
            out = self.prior_head(torch.cat([seq, img_feature], -1))
        else:
            encoded = self._transformer_encode(memory, pad_mask=None)
            last_feat = encoded[None, -1]
            if img_feature.shape[0] > 1:
                last_feat = last_feat.expand((img_feature.shape[0], *last_feat.shape[1:]))
            out = self.prior_head(torch.cat([last_feat, img_feature], -1))

            if img_feature.shape[0] == 1:
                new_memory = torch.cat([memory, img_feature], dim=0)

        if update_hidden:
            self._memory = new_memory

        return out

    def loss(self, output, inputs):
        losses = AttrDict()
        B, T, C = output.prior.shape

        mse_loss = torch.nn.MSELoss()
        nll_loss = torch.nn.NLLLoss()

        losses.rec_mse = mse_loss(output.reconstruction, inputs.actions.view(B * T, *inputs.actions.shape[2:]))
        losses.prior = nll_loss(output.prior.reshape(B * T, C), inputs.skills.argmax(dim=-1).reshape(B * T))

        losses.total = losses.rec_mse + losses.prior
        return losses

    def reset_hidden_state(self):
        self._memory = None