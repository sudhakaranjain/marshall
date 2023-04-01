from typing import List

import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, image_input_size: int, image_channels: int):
        """
        Initializes the decoder model that contains both image_decoder and text_decoder.

        :param hidden_size: dimension for projection embedding
        :param patch_size: dimension for each patch created from the image
        :param image_input_size: dimension for the image in the batch
        :param image_channels: no. of channels that the images have
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.image_input_size = image_input_size
        self.image_channels = image_channels
        self.image_decoder = ImageDecoder(hidden_size, patch_size, image_input_size, image_channels)
        self.text_decoder = TextDecoder(hidden_size)

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
        self.linear_projection = nn.Linear(hidden_size, (patch_size // 2) ** 2)
        self.conv2d_t1 = nn.ConvTranspose2d(1, 32, 3, stride=1)
        self.conv2d_t2 = nn.ConvTranspose2d(32, 16, 3, stride=1)
        self.conv2d_t3 = nn.ConvTranspose2d(16, 8, 3, stride=1)
        self.conv2d_t4 = nn.ConvTranspose2d(8, image_channels, 3, stride=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Gets the decoded batch containing either reconstructed images.

        :param features: features obtained from Marshall that needs to be decoded into images
        """

        x = self.linear_projection(features)
        x = self.conv2d_t1(x.reshape(-1, 1, (self.patch_size // 2), (self.patch_size // 2)))
        x = self.conv2d_t2(x)
        x = self.conv2d_t3(x)
        x = self.conv2d_t4(x)
        x = x.reshape(*features.shape[:2], *x.shape[1:])
        x = x.transpose(1, 2)
        return x.reshape(*x.shape[:2], self.image_input_size, self.image_input_size)


# TODO: Implement the text decoder to reconstruct the captions
class TextDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 30522, max_seq_len: int = 512):

        super(TextDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, text_batch: torch.Tensor, ) -> torch.Tensor:
