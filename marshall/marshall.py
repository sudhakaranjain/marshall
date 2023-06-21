from typing import Any, List, Union

import torch
import torch.nn as nn

from marshall.ema import EMA


class MultiModalEncoder(nn.Module):
    def __init__(self, config):
        """
        Initializes a Multimodal Encoder with modality specific feature extractor and a fusion layer w.r.t the params
        in the config.

        :param config: Config file containing various hyper parameters needed for initializing the MMEncoder
        """
        super(MultiModalEncoder, self).__init__()
        self.hidden = config.model.hidden_dim

        # Fusion layer
        self.config = config
        self.fusion_layer = []
        for i in range(self.config.model.n_layers):
            if i < self.config.model.n_layers - 1:
                self.fusion_layer.append(nn.Sequential(
                    nn.TransformerEncoderLayer(d_model=self.hidden, nhead=self.config.model.attn_heads,
                                               batch_first=True),
                    nn.LayerNorm(self.hidden),
                ))
            else:
                self.fusion_layer.append(nn.Sequential(
                    nn.TransformerEncoderLayer(d_model=self.hidden, nhead=self.config.model.attn_heads,
                                               batch_first=True),
                    nn.LayerNorm(self.hidden),
                    nn.Linear(self.hidden, self.hidden),
                    nn.Tanh(),
                ))
        self.fusion_layer = nn.ModuleList(self.fusion_layer)

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass: gets the features w.r.t the modality and passes through the fusion layer.

        :param x: Input batch
        :returns: representation vectors of the input batch
        """
        for i in range(self.config.model.n_layers):
            x = self.fusion_layer[i](x)
        return x


class Marshall(nn.Module):
    def __init__(self, student_model, device, config, **kwargs):
        super(Marshall, self).__init__()
        self.config = config
        self.embed_dim = self.config.model.hidden_dim
        self.student_model = student_model
        self.__dict__.update(kwargs)

        self.ema = EMA(self.student_model, self.config, device=device)  # EMA acts as the teacher
        self.ema_decay = self.config.model.ema_decay
        self.ema_end_decay = self.config.model.ema_end_decay
        self.ema_anneal_end_step = self.config.model.ema_anneal_end_step

        self.text_regression_head = self._build_regression_head(modality='text')
        self.vision_regression_head = self._build_regression_head(modality='vision')

    def _build_regression_head(self, modality: str) -> Any:
        """
        Construct the regression head consisting of linear and activation layers.
        Each modality might have its own regression block.

        :param modality: Type of data
        :returns: A nn.Module layer or block of layers
        """
        if modality == 'text':
            return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2),
                                 nn.GELU(),
                                 nn.BatchNorm1d(self.embed_dim // 2),
                                 nn.Linear(self.embed_dim // 2, self.embed_dim))

        if modality == 'vision':
            return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2),
                                 nn.GELU(),
                                 nn.BatchNorm1d(self.embed_dim // 2),
                                 nn.Linear(self.embed_dim // 2, self.embed_dim))

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.student_model)

    def forward(self, input_modality: str, input_batch, reference_batch=None, **kwargs):
        """
        Marshall forward method.

        :param input_modality: Modality of the input batch ('text'/'vision')
        :param input_batch: Input batch of tokens of either of the modality which is encoded during training.
        :param reference_batch: Reference batch of tokens is of the modality != that of the input batch.
        :returns: Either encoder outputs or a tuple of encoder + EMA outputs
        """
        # model forward in online mode (student)
        # fetch the last token embedding output
        x = self.student_model(input_batch)
        if reference_batch is None:
            return x

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(reference_batch)[:, -1, :]  # fetch the last token embedding output

        # TODO: Test applying normalization to embeddings
        x_computed = x.clone()
        x_computed[:, -1, :] = self.text_regression_head(x[:, -1, :]) if input_modality == 'text' else \
            self.vision_regression_head(x[:, -1, :])

        # returning x with  all token embedding output, y with last token embedding output
        return x_computed, y
