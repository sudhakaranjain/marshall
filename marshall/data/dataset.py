import json
import os
import random
from typing import Dict, List, Union

import numpy as np
import torch
import omegaconf
from omegaconf.dictconfig import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer

import io
import urllib

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()

# Get config file
config = omegaconf.OmegaConf.load('../marshall/configs.yaml')
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model.text.pretrained_model)


class SingleClassDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 caption_path: str,
                 config: DictConfig):
        """
        Initialize the SingleClassDataset.

        :param dataset_path: path to the folder containing the images
        :param caption_path: path to json containing the captions and their corresponding image file paths
        :param config: config file for model and dataset
        :param tokenizer: tokenizer to tokenize text captions
        :param max_length: parameter to set maximum length of tokens
        """
        self.dataset_path = dataset_path
        self.config = config
        self.tokenizer = tokenizer

        with open(caption_path, "r") as f:
            self.img_paths_and_captions = json.load(f)

        # Only considering the first caption from list of captions for a given image
        self.all_captions = [path["captions"][0] for path in self.img_paths_and_captions]

    def __len__(self) -> int:
        """Returns the len of the Dataset"""
        return len(self.img_paths_and_captions)

    def __getitem__(self, idx: int) -> Dict[str, Union[Dict[str, List[int]], torch.tensor]]:
        """
        Returns the instance at index 'idx'. The instance is a Dict containing the image as tensor and corresponding
        caption as tokens.
        """
        img = Image.open(os.path.join(self.dataset_path, self.img_paths_and_captions[idx]['file_name']))
        return {"image": to_tensor(img.resize((self.config.dataset.input_size, self.config.dataset.input_size))),
                "caption": self.all_captions[idx]}

    @staticmethod
    def collate_fn(batch):
        images = []
        captions = []
        for d in batch:
            images.append(d['image'])
            captions.append(d['caption'])

        x = random.uniform(0, 1)
        if x <= 0.5:
            return {"input_modality": 'vision',
                    "student": torch.stack(images),
                    "reference": tokenizer(captions, max_length=None, truncation=True, padding=True,
                                           return_tensors='pt', return_attention_mask=False)['input_ids']}
        else:
            return {"input_modality": 'text',
                    "student": tokenizer(captions, max_length=None, truncation=True, padding=True,
                                         return_tensors='pt', return_attention_mask=False)['input_ids'],
                    "reference": torch.stack(images)}


class SBUDataset(Dataset):
    def __init__(self, config: DictConfig):
        """
        Initialize the SingleClassDataset.

        :param dataset_path: path to the folder containing the images
        :param caption_path: path to json containing the captions and their corresponding image file paths
        :param config: config file for model and dataset
        :param tokenizer: tokenizer to tokenize text captions
        :param max_length: parameter to set maximum length of tokens
        """
        self.cc_dataset = load_dataset("facebook/pmd", "sbu", split="train", use_auth_token=True)
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the len of the Dataset"""
        return len(self.cc_dataset)

    def fetch_single_image(self, image_data, timeout=None, retries=0):
        image_url = image_data["image_url"]
        image = image_data["image"]
        text = image_data["text"]
        if image is not None:
            # check if number of channels is 3
            if np.asarray(image).shape[-1] == 3:
                image = to_tensor(image.resize((self.config.dataset.input_size, self.config.dataset.input_size)))
            else:
                image = None
            return image, text

        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = Image.open(io.BytesIO(req.read()))
                    # check if number of channels is 3
                    if np.asarray(image).shape[-1] == 3:
                        image = to_tensor(image.resize((self.config.dataset.input_size,
                                                        self.config.dataset.input_size)))
                    else:
                        image = None
                break
            except Exception:
                image = None
        return image, text

    def __getitem__(self, idx: int) -> Dict[str, Union[Dict[str, List[int]], torch.tensor]]:
        """
        Returns the instance at index 'idx'. The instance is a Dict containing the image as tensor and corresponding
        caption as tokens.
        """
        image_data = self.cc_dataset[idx]
        img, text = self.fetch_single_image(image_data)
        return {"image": img, "caption": text}

    @staticmethod
    def collate_fn(batch):
        images = []
        captions = []
        for d in batch:
            if d['image'] is not None:
                images.append(d['image'])
                captions.append(d['caption'])

        x = random.uniform(0, 1)
        if x <= 0.5:
            return {"input_modality": 'vision',
                    "student": torch.stack(images),
                    "reference": tokenizer(captions, max_length=None, truncation=True, padding=True,
                                           return_tensors='pt', return_attention_mask=False)['input_ids']}
        else:
            return {"input_modality": 'text',
                    "student": tokenizer(captions, max_length=None, truncation=True, padding=True,
                                         return_tensors='pt', return_attention_mask=False)['input_ids'],
                    "reference": torch.stack(images)}
