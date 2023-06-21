import argparse
import os

import omegaconf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import SingleClassDataset
from marshall import Marshall, MultiModalEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_path):
    state_dict = torch.load(model_path, map_location=torch.device(device))
    return state_dict['student']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Marshall Training", add_help=True)

    parser.add_argument("-d", "--dataset-path", type=str, default="./datasets", help="Path to the dataset folder")

    args = parser.parse_args()

    # Get config file
    config = omegaconf.OmegaConf.load('marshall/configs.yaml')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.pretrained_model)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # load model
    model = MultiModalEncoder(config)
    model.load_state_dict(
        load_checkpoint(model_path=os.path.join(config.train.checkpoint_path, 'marshall_%d.pth' % 20)))
    model.eval().to(device)

    val_dataset = SingleClassDataset(images_path=os.path.join(args.dataset_path, "class_3_val"),
                                     file_paths="class_3_val.json", tokenizer=tokenizer, config=config)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size,
                                collate_fn=SingleClassDataset.collate_fn, shuffle=False, drop_last=True,
                                num_workers=10)
    marshall = Marshall(model, device, config)
    with torch.no_grad():
        for batch in val_dataloader:
            # x = torch.stack([a.sum(dim=1) for a in model(batch['input_modality'], batch['input'])[-3:]]).sum(dim=0)
            x = model(batch['input_modality'], batch['input'])[-1].sum(dim=1)
            reference_modality = 'text' if batch['input_modality'] == 'vision' else 'vision'
            # y = torch.stack([a.sum(dim=1) for a in model(reference_modality, batch['reference'])[-3:]]).sum(dim=0)
            y = model(reference_modality, batch['reference'])[-1].sum(dim=1)
            break

    print('SIMILARITY BETWEEN MODALITY: ', cos(x, y))
    r = torch.randperm(x.size(0))
    print('SIMILARITY WITH RANDOMIZED VALUES', cos(x[r], y))
    print(x)
    print(y)
