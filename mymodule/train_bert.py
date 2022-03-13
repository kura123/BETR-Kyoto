import torch

import pytorch_lightning as pl
from transformers import BertForMaskedLM


class BertForMaskedLM_pl(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    
    def training_step(self, batch, batch_idx):
        output = self.bert_mlm(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.bert_mlm(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        