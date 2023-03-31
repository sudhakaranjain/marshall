import math
from typing import Tuple

from einops.layers.torch import Rearrange
from torch.autograd import Variable
from transformers import AutoTokenizer
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, pretrained_model: str, patch_size: int, hidden_size: int):
        """
        Initializes the encoder model that contains both image_encoder and text_encoder.

        :param pretrained_model: pretrained model (found in huggingface) to be used as a tokenizer.
        :param patch_size: dimension for each patch created from the image
        :param hidden_size: dimension for projection embedding
        """
        super(Encoder, self).__init__()

        self.image_encoder = ImageEncoder(patch_size=patch_size, hidden_size=hidden_size)
        self.text_encoder = TextEncoder(pretrained_model=pretrained_model,
                                        hidden_size=hidden_size)

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
    def __init__(self, patch_size: int, hidden_size: int):
        """
        Initializes the pretrained resnet18 model to be used as feature extractor.

        :param patch_size: dimension for each patch created from the image
        :param hidden_size: dimension for projection embedding
        """
        super(ImageEncoder, self).__init__()

        self.patch_n_flatten_layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.patch_projection = nn.Linear(patch_size * patch_size * 3, hidden_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Gets the representation batch i.e. the output for the input batch of images.

        :param image: input batch images that needs to be processed (patching and feature extraction)
        """
        x = self.patch_n_flatten_layer(image)
        return self.patch_projection(x)


# TODO: Implemented but should be verified once again, and doc strings need to be added
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model, hidden_size: int, vocab_size: int = 30522, max_seq_len: int = 512):
        """
        :param hidden_size: dimension for projection embedding
        """
        super(TextEncoder, self).__init__()
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
