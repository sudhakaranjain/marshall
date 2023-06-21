import argparse
import os
from typing import List, Tuple

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from marshall.data.dataset import SingleClassDataset
from marshall import Marshall, MultiModalEncoder, Encoder, Decoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MarshallTrainer(pl.LightningModule):
    def __init__(self, config):
        super(MarshallTrainer, self).__init__()

        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.student = MultiModalEncoder(config).to(device)
        self.marshall = Marshall(student_model=self.student, device=device, config=config).to(device)
        self.l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input_modality, student_batch, reference_batch) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # encoding the modalities with modal specific encoders
        student_input, reference_input = self.encoder(input_modality, student_batch, reference_batch)
        # getting the output representations with the (common) modality agnostic model
        student_out, reference_out = self.marshall(input_modality, student_input, reference_input)
        # reconstructing the input from all but last token embeddings
        reconstructed = self.decoder(input_modality, student_out[:, :-1, :])

        return student_out, reference_out, reconstructed

    def training_step(self, batch, batch_idx) -> torch.tensor:
        # student_out contains all token embedding outputs
        # reference_out contains last token embedding output
        student_out, reference_out, reconstructed = self(batch['input_modality'], batch['student'], batch['reference'])

        # reconstruction loss
        recon_loss = self.l1_loss(reconstructed.float(), batch['student'].float()) \
            if batch['input_modality'] == 'vision' else self.ce_loss(reconstructed.float(), batch['student'].long())

        # loss for minimizing the distance between last token embedding representation
        multi_modal_loss = self.l1_loss(student_out[:, -1, :].float(), reference_out.float()) + \
            (1 - F.cosine_similarity(student_out[:, -1, :].float(), reference_out.float()).mean())

        # logging
        self.logger.experiment.add_scalar("loss/train_loss", multi_modal_loss + recon_loss, self.current_epoch)

        return multi_modal_loss + recon_loss

    def on_train_batch_end(self, *args, **kwargs):
        self.marshall.ema_step()

    def validation_step(self, batch, batch_idx):
        # student_out contains all token embedding outputs
        # reference_out contains last token embedding output
        student_out, reference_out, reconstructed = self(batch['input_modality'], batch['student'], batch['reference'])

        # reconstruction loss
        recon_loss = self.l1_loss(reconstructed.float(), batch['student'].float()) \
            if batch['input_modality'] == 'vision' else self.ce_loss(reconstructed.float(), batch['student'].long())

        # loss for minimizing the distance between last token embedding representation
        multi_modal_loss = self.l1_loss(student_out[:, -1, :].float(), reference_out.float()) + \
                           (1 - F.cosine_similarity(student_out[:, -1, :].float(), reference_out.float()).mean())

        # logging
        self.logger.experiment.add_scalar("loss/val_loss", multi_modal_loss + recon_loss, self.current_epoch)

        return multi_modal_loss + recon_loss

    def training_epoch_end(self, outputs: List):
        if (self.trainer.current_epoch + 1) % self.config.train.save_ckpt_freq == 0:
            torch.save({'student': self.marshall.student_model.state_dict(), 'encoder': self.encoder.state_dict(),
                        'decoder': self.decoder.state_dict()},
                       os.path.join(self.config.train.checkpoint_path,
                                    'marshall_%d.pth' % (self.trainer.current_epoch + 1)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.marshall.parameters(), self.config.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.config.optimizer.lr_decay)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Marshall Training", add_help=True)

    parser.add_argument("-d", "--dataset-path", type=str, default="../datasets", help="Path to the dataset folder")

    args = parser.parse_args()

    # Get config file
    config = omegaconf.OmegaConf.load('../marshall/configs.yaml')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.text.pretrained_model)

    # Initialize Dataset
    train_dataset = SingleClassDataset(dataset_path=os.path.join(args.dataset_path, "train2017"),
                                       caption_path=os.path.join(args.dataset_path, "train_captions.json"),
                                       tokenizer=tokenizer, config=config)
    val_dataset = SingleClassDataset(dataset_path=os.path.join(args.dataset_path, "val2017"),
                                     caption_path=os.path.join(args.dataset_path, "val_captions.json"),
                                     tokenizer=tokenizer, config=config)

    # Initialize Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                  collate_fn=SingleClassDataset.collate_fn, shuffle=True, drop_last=True,
                                  num_workers=10)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size,
                                collate_fn=SingleClassDataset.collate_fn, shuffle=False, drop_last=True,
                                num_workers=10)

    # Initialize Trainer and train the model
    model = MarshallTrainer(config)
    logger = TensorBoardLogger('tb_logs', name='marshall_train_logs')
    if torch.cuda.is_available():
        trainer = Trainer(gpus=-1, auto_select_gpus=True, max_epochs=config.train.epochs, gradient_clip_val=1.0,
                          logger=logger)
    else:
        trainer = Trainer(max_epochs=config.train.epochs, gradient_clip_val=1.0, logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)
