import json
import os
import random
from typing import Any, Dict, List, Union

import torch
import omegaconf
from omegaconf.dictconfig import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer

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
