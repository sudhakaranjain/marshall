from typing import Any, List, Union
import math

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops.layers.torch import Rearrange
from transformers import AutoModel, AutoTokenizer

from marshall.ema import EMA


class ImageFeatureExtractor(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, freeze: bool = True):
        """
        Initializes the pretrained resnet18 model to be used as feature extractor.

        :param patch_size: dimension for each patch created from the image
        :param hidden_size: dimension for projection embedding
        :param freeze: Whether to freeze the model so that it is not updated during backprop
        """
        super(ImageFeatureExtractor, self).__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.image_embedding = nn.Linear(self.model.fc.in_features, hidden_size)
        self.model.fc = nn.Identity()
        self.patch_layer = Rearrange('b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_size, p2=patch_size)

    def forward(self, image: torch.Tensor, average: bool = True) -> torch.Tensor:
        """
        Gets the representation batch i.e. the output for the input batch of images.

        :param image: input batch images that needs to be processed (patching and feature extraction)
        :param average: Whether to average the patch embeddings or pass them separately to attention
        """
        x = self.patch_layer(image)
        features = torch.stack([self.model(y) for y in x])
        if average:
            return torch.mean(self.image_embedding(features), 1)
        return self.image_embedding(features)


class TextFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model, freeze=True):
        """
        Initializes the pretrained model to be used as feature extractor.
        Note: Tokenizer and the model should match. So ensure they use the same model params.

        :param pretrained_model: pretrained model (found in huggingface) to be used as feature extractor.
        :param freeze: Whether to freeze the model so that it is not updated during backprop
        """
        super(TextFeatureExtractor, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text_batch: torch.Tensor, pooling: bool = True) -> torch.Tensor:
        """
        Gets the representation batch i.e. the pooled output for the input batch  of tokens.

        :param text_batch:
        :param pooling:
        """
        model_output = self.model(text_batch)
        if pooling:
            return model_output.pooler_output
        return model_output.last_hidden_state


class ImageEmbedder(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int):
        """
        Initializes the pretrained resnet18 model to be used as feature extractor.

        :param patch_size: dimension for each patch created from the image
        :param hidden_size: dimension for projection embedding
        """
        super(ImageEmbedder, self).__init__()

        self.patch_n_flatten_layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.patch_projection = nn.Linear(patch_size * patch_size * 3, hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Gets the representation batch i.e. the output for the input batch of images.

        :param image: input batch images that needs to be processed (patching and feature extraction)
        """
        x = self.patch_n_flatten_layer(image)
        return self.patch_projection(x)


class TextEmbedder(nn.Module):
    def __init__(self, pretrained_model, hidden_size: int, vocab_size: int = 30522, max_seq_len: int = 512):
        """
        :param hidden_size: dimension for projection embedding
        """
        super(TextEmbedder, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Positional Encoding:
        # create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_seq_len, hidden_size)
        for pos in range(max_seq_len):
            for i in range(0, hidden_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / hidden_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / hidden_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, text_batch: torch.Tensor, ) -> torch.Tensor:
        """
        Gets the representation batch i.e. the pooled output for the input batch  of tokens.

        :param text_batch:
        """
        x = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")['input_ids']
        x = self.embed(x)
        x = x * math.sqrt(self.hidden_size)  # make embeddings relatively larger
        seq_len = x.size(1)  # add constant to embedding
        return x + Variable(self.pe[:, :seq_len], requires_grad=False)


class MultiModalEncoder(nn.Module):
    def __init__(self, config):
        """
        Initializes a Multimodal Encoder with modality specific feature extractor and a fusion layer w.r.t the params
        in the config.

        :param config: Config file containing various hyper parameters needed for initializing the MMEncoder
        """
        super(MultiModalEncoder, self).__init__()
        self.hidden = config.model.hidden_dim
        # self.image_embedder = ImageFeatureExtractor(patch_size=config.dataset.patch_size,
        #                                                      hidden_size=config.model.hidden_dim,
        #                                                      freeze=config.model.vision.freeze)
        # self.caption_embedder = TextFeatureExtractor(pretrained_model=config.model.text.pretrained_model,
        #                                                       freeze=config.model.text.freeze)

        self.image_embedder = ImageEmbedder(patch_size=config.dataset.patch_size, hidden_size=config.model.hidden_dim)
        self.caption_embedder = TextEmbedder(pretrained_model=config.model.text.pretrained_model,
                                             hidden_size=config.model.hidden_dim)

        # Fusion layer
        # TODO: Experiment with different fusion layer
        self.fusion_layer_type = config.model.fusion_layer
        self.config = config
        if self.fusion_layer_type == 'transformer':
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
        elif self.fusion_layer_type == 'mlp':
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.hidden, self.hidden * 2),
                nn.GELU(),
                nn.Linear(self.hidden * 2, self.hidden)
            )

    def forward(self, modality: str, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass: gets the features w.r.t the modality and passes through the fusion layer.

        :param modality: Type of org. input data supports ['text', 'vision']
        :param x: Input batch
        :returns: representation vectors of the input batch
        """
        is_pooled_features = True if self.fusion_layer_type == 'mlp' else False
        features = self.image_embedder(x) if modality == 'vision' \
            else self.caption_embedder(x)

        if self.fusion_layer_type == 'transformer':
            output = []
            for i in range(self.config.model.n_layers):
                features = self.fusion_layer[i](features)
                output.append(features)
            return output
        elif self.fusion_layer_type == 'mlp':
            return self.fusion_layer(features)


class Marshall(nn.Module):
    def __init__(self, encoder, device, config, **kwargs):
        super(Marshall, self).__init__()
        self.config = config
        self.embed_dim = self.config.model.hidden_dim
        self.student_encoder = encoder
        self.__dict__.update(kwargs)

        self.ema = EMA(self.student_encoder, self.config, device=device)  # EMA acts as the teacher
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

        if modality in ['vision']:
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
            self.ema.step(self.student_encoder)

    # TODO: modify this whole below function to adapt it with our custom mmBERT
    def forward(self, input_modality: str, input_batch, reference_batch=None, **kwargs):
        """
        Marshall forward method.

        :param input_modality: Modality of the input batch ('text'/'vision')
        :param input_batch: Input batch of tokens of either of the modality which is encoded during training.
        :param reference_batch: Reference batch of tokens is of the modality != that of the input batch.
        :returns: Either encoder outputs or a tuple of encoder + EMA outputs
        """
        # model forward in online mode (student)
        # fetch the last layer outputs
        x = self.student_encoder(input_modality, input_batch)
        x = x[-1].sum(dim=1)
        if reference_batch is None:
            return x

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            reference_modality = 'text' if input_modality == 'vision' else 'vision'
            y = self.ema.model(reference_modality, reference_batch)  # fetch the last transformer layers outputs
            y = y[-self.config.model.average_top_k_layers:]  # take the last k transformer layers

            # Follow the same layer normalization procedure for text and vision
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)
            y = torch.sum(y, dim=1)
            if self.config.model.normalize_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

        x = self.text_regression_head(x) if input_modality == 'text' else self.vision_regression_head(x)

        return x[:, :128], y[:, :128]
