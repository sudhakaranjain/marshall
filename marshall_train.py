import argparse
import os
from typing import List

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import SingleClassDataset
from marshall import Marshall, MultiModalEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MarshallTrainer(pl.LightningModule):
    def __init__(self, config):
        super(MarshallTrainer, self).__init__()

        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.student = MultiModalEncoder(config).to(device)
        self.marshall = Marshall(encoder=self.student, device=device, config=config).to(device)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, input_modality, input_batch, reference_batch):
        return self.marshall(input_modality, input_batch, reference_batch)

    def training_step(self, batch, batch_idx) -> torch.tensor:
        x, y = self(batch['input_modality'], batch['input'], batch['reference'])
        loss = self.criterion(x.float(), y.float()) + (1 - F.cosine_similarity(x.float(), y.float()).mean())

        # logging
        self.logger.experiment.add_scalar("loss/train_loss", loss, self.current_epoch)

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.marshall.ema_step()

    def validation_step(self, batch, batch_idx):
        x, y = self(batch['input_modality'], batch['input'], batch['reference'])
        val_loss = self.criterion(x.float(), y.float()) + (1 - F.cosine_similarity(x.float(), y.float()).mean())

        # logging
        self.logger.experiment.add_scalar("loss/val_loss", val_loss, self.current_epoch)

        return val_loss

    def training_epoch_end(self, outputs: List):
        if (self.trainer.current_epoch+1) % self.config.train.save_ckpt_freq == 0:
            torch.save({'student': self.marshall.student_encoder.state_dict()},
                       os.path.join(self.config.train.checkpoint_path, 'marshall_%d.pth' % (self.trainer.current_epoch+1)))

    def configure_optimizers(self):
        return torch.optim.Adam(self.marshall.parameters(), self.config.optimizer.lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Marshall Training", add_help=True)

    parser.add_argument("-d", "--dataset-path", type=str, default="./datasets", help="Path to the dataset folder")

    args = parser.parse_args()

    # Get config file
    config = omegaconf.OmegaConf.load('marshall/configs.yaml')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.pretrained_model)

    # Initialize Dataset
    train_dataset = SingleClassDataset(images_path=os.path.join(args.dataset_path, "class_3_train"),
                                       file_paths="class_3_train.json", tokenizer=tokenizer, config=config)
    val_dataset = SingleClassDataset(images_path=os.path.join(args.dataset_path, "class_3_val"),
                                     file_paths="class_3_val.json", tokenizer=tokenizer, config=config)

    # Initialize Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                  collate_fn=SingleClassDataset.collate_fn, shuffle=True, drop_last=True,
                                  num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size,
                                collate_fn=SingleClassDataset.collate_fn, shuffle=False, drop_last=True,
                                num_workers=10)
    # TODO: Tested till the above line and it works

    # Initialize Trainer and train the model
    model = MarshallTrainer(config)
    logger = TensorBoardLogger('tb_logs', name='marshall_train_logs')
    if torch.cuda.is_available():
        trainer = Trainer(gpus=-1, auto_select_gpus=True, max_epochs=config.train.epochs, gradient_clip_val=1.0,
                          logger=logger)
    else:
        trainer = Trainer(max_epochs=config.train.epochs, gradient_clip_val=1.0, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
