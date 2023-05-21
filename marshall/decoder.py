from typing import List

import omegaconf
import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        """
        Initializes the decoder model that contains both image_decoder and text_decoder.

        :param cfg: contains configuration data
        """
        super(Decoder, self).__init__()

        self.hidden_size = cfg.model.hidden_dim
        self.patch_size = cfg.dataset.patch_size
        self.image_input_size = cfg.dataset.input_size
        self.image_channels = cfg.dataset.in_channels

        self.image_decoder = ImageDecoder(self.hidden_size, self.patch_size, self.image_input_size, self.image_channels)
        self.text_decoder = TextDecoder(self.hidden_size)

    def forward(self, modality: str, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Gets the decoded batch containing either reconstructed images or text.

        :param modality: Modality of the input batch ('text'/'vision')
        :param features: features obtained from Marshall that needs to be decoded into either images or tokens
        """
        if modality == 'vision':
            return self.image_decoder(features)
        elif modality == 'text':
            return self.text_decoder(features)


class ImageDecoder(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, image_input_size: int, image_channels: int):
        """
        Initializes the image_decoder model.

        :param hidden_size: dimension for projection embedding
        :param patch_size: dimension for each patch created from the image
        :param image_input_size: dimension for the image in the batch
        :param image_channels: no. of channels that the images have
        """
        super(ImageDecoder, self).__init__()
        self.patch_size = patch_size
        self.image_input_size = image_input_size
        self.linear_projection = nn.Linear(hidden_size, hidden_size)
        self.conv2d_t1 = nn.ConvTranspose2d(12, 16, 7, stride=1)
        self.conv2d_t2 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.conv2d_t3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv2d_t4 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.conv2d_t5 = nn.ConvTranspose2d(8, image_channels, 2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Gets the decoded batch containing reconstructed images.

        :param features: features obtained from Marshall that needs to be decoded into images
        """

        # features is of shape: [batch_size, 1, hidden_dim]
        # projecting it to shape: [batch_size, 1, hidden_dim]
        x = self.linear_projection(features)
        x = self.relu(x)
        # reshaping to [batch_size, deduced_channels: 12 (for hidden_dim: 768), 8, 8]
        x = x.reshape(features.shape[0], -1, 8, 8)
        x = self.conv2d_t1(x)
        x = self.relu(x)
        x = self.conv2d_t2(x)
        x = self.relu(x)
        x = self.conv2d_t3(x)
        x = self.relu(x)
        x = self.conv2d_t4(x)
        x = self.relu(x)
        x = self.conv2d_t5(x)
        x = self.relu(x)
        return x  # resulting shape: [batch_size, 3, image_input_size, image_input_size]


# TODO: Implemented but should be verified once again
class TextDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 30522, max_seq_len: int = 512):
        """
        :param hidden_size: dimension for projection embedding
        :param vocab_size: vocab size of the tokenizer used
        :param max_seq_len: maximum number of tokens in the input sequence
        """
        super(TextDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, vocab_size),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Gets the decoded batch containing reconstructed text.

        :param features: features obtained from Marshall that needs to be decoded into text
        """

        # features is of shape: [batch_size, sequence_length, hidden_dim]
        x = self.mlp(features)  # result shape: [batch_size, sequence_length, vocab_size]
        # swapping dimensions for computation of CE loss
        return x.transpose(1, 2)  # returns tokens of shape: [batch_size, vocab_size, sequence_length]
