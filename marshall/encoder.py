import math
from typing import Tuple

import omegaconf
from einops.layers.torch import Rearrange
from torch.autograd import Variable
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        """
        Initializes the encoder model that contains both image_encoder and text_encoder.

        :param cfg: contains configuration data

        """
        self.pretrained_model = cfg.model.text.pretrained_model
        self.image_input_size = cfg.dataset.input_size
        self.patch_size = cfg.dataset.patch_size
        self.hidden_size = cfg.model.hidden_dim
        self.dropout = cfg.model.dropout
        super(Encoder, self).__init__()

        self.image_encoder = ImageEncoder(image_input_size=self.image_input_size, patch_size=self.patch_size,
                                          hidden_size=self.hidden_size, dropout=self.dropout)
        self.text_encoder = TextEncoder(hidden_size=self.hidden_size, dropout=self.dropout)

    def forward(self, modality: str, input_batch: torch.Tensor, reference_batch: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                        torch.Tensor]:
        """
        Gets the encoded batch i.e. the input batch of images/text that will be fed to Marshall.

        :param modality: Modality of the input batch ('text'/'vision')
        :param input_batch: input batch that needs to be processed (patching or tokenizing)
        :param reference_batch: reference batch that needs to be processed (patching or tokenizing)
        """
        if modality == 'vision':
            return self.image_encoder(input_batch), self.text_encoder(reference_batch)
        elif modality == 'text':
            return self.text_encoder(input_batch), self.image_encoder(reference_batch)


class ImageEncoder(nn.Module):
    def __init__(self, image_input_size: int, patch_size: int, hidden_size: int, dropout: int):
        """
        Initializes the pretrained resnet18 model to be used as feature extractor.

        :param image_input_size: dimension for the image in the batch
        :param patch_size: dimension for each patch created from the image
        :param hidden_size: dimension for projection embedding
        """
        super(ImageEncoder, self).__init__()

        no_of_patches = (image_input_size // patch_size) ** 2
        self.patch_n_flatten_layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.patch_projection = nn.Linear(patch_size * patch_size * 3, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, no_of_patches, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Gets the representation batch i.e. the output for the input batch of images.

        :param image: input batch images that needs to be processed (patching and feature extraction)
        """
        x = self.patch_n_flatten_layer(image)
        x = self.patch_projection(x)
        x += self.pos_embedding
        return self.dropout(x)


# TODO: Implemented but should be verified once again
class TextEncoder(nn.Module):
    def __init__(self, hidden_size: int, dropout: int, vocab_size: int = 30522,
                 max_seq_len: int = 512):
        """
        :param hidden_size: dimension for projection embedding
        :param dropout: dropout probability
        :param vocab_size: vocab size of the tokenizer used
        :param max_seq_len: maximum number of tokens in the input sequence
        """
        super(TextEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

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

        :param text_batch: input text batch that needs to be processed (linear projection and positional encoding)
        """
        x = self.embed(text_batch)
        x = x * math.sqrt(self.hidden_size)  # make embeddings relatively larger
        seq_len = x.size(1)  # add constant to embedding
        x += Variable(self.pe[:, :seq_len], requires_grad=False)
        return self.dropout(x)
