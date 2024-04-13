import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy, BinaryMatthewsCorrCoef
from torchmetrics.classification import MultilabelAccuracy, MultilabelMatthewsCorrCoef
import pytorch_lightning as pl
from transformers import AutoModelForAudioClassification

from config import Config


class Wav2Vec2Model(pl.LightningModule):
    def __init__(
            self,
            config=Config
    ):
        super(Wav2Vec2Model, self).__init__()
        self.config = config
        self.n_outputs = self.config.N_OUTPUT_CLASS
        self.learning_rate = self.config.LEARNING_RATE

        self.model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=self.n_outputs)
        layer_to_unfreeze = self.config.LAYER_TO_UNFREEZE_FROM
        for param in self.model.parameters():
            param.requires_grad = True
        for name, param in self.model.named_parameters():
            if f"wav2vec2.encoder.layers.{layer_to_unfreeze}" in name:
                break
            else:
                param.requires_grad = False

        if self.n_outputs > 1:
            self.accuracy = MultilabelAccuracy(num_labels=self.n_outputs)
            self.mcc = MultilabelMatthewsCorrCoef(num_labels=self.n_outputs)
        else:
            self.accuracy = BinaryAccuracy()
            self.mcc = BinaryMatthewsCorrCoef()

        self.TRAIN_ACTUAL = []
        self.TRAIN_PRED = []
        self.VAL_ACTUAL = []
        self.VAL_PRED = []
        self.TEST_ACTUAL = []
        self.TEST_PRED = []


    def forward(self, input_data):
        out = self.model(input_data).logits
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        label_loss = []
        for i, label in enumerate(self.config.LABEL_COLUMNS):
            pos_weight = self.config.get_pos_weight(i)
            loss = F.binary_cross_entropy_with_logits(logits[:, i], y[:, i], pos_weight=pos_weight)
            label_loss.append(loss)
            self.log(f"train_{label}_loss", loss)

        binary_loss = F.binary_cross_entropy_with_logits(logits[:, -1], y[:, -1], pos_weight=self.config.get_pos_weight(-1))
        self.log("train_binary_loss", binary_loss)

        loss = binary_loss * self.config.BINARY_WEIGHT + torch.mean(torch.tensor(label_loss)) * (1 - self.config.BINARY_WEIGHT)
        self.log("train_loss", loss)

        self.TRAIN_ACTUAL.append(y.long())
        self.TRAIN_PRED.append(logits)

        return loss
    
    def on_train_epoch_end(self):
        actual = torch.concat(self.TRAIN_ACTUAL, dim=0)
        preds = torch.concat(self.TRAIN_PRED, dim=0)
        
        acc = self.accuracy(preds, actual)
        mcc = self.mcc(preds, actual)

        self.log_dict({
            "train_acc": acc,
            "train_mcc": mcc
        }, prog_bar=True, on_epoch=True, on_step=False)

        self.TRAIN_ACTUAL.clear()
        self.TRAIN_PRED.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        label_loss = []
        for i, label in enumerate(self.config.LABEL_COLUMNS):
            pos_weight = self.config.get_pos_weight(i)
            loss = F.binary_cross_entropy_with_logits(logits[:, i], y[:, i], pos_weight=pos_weight)
            label_loss.append(loss)
            self.log(f"val_{label}_loss", loss)

        binary_loss = F.binary_cross_entropy_with_logits(logits[:, -1], y[:, -1], pos_weight=self.config.get_pos_weight(-1))
        self.log("val_binary_loss", binary_loss)

        loss = binary_loss * self.config.BINARY_WEIGHT + torch.mean(torch.tensor(label_loss)) * (1 - self.config.BINARY_WEIGHT)
        self.log("val_loss", loss)

        self.VAL_ACTUAL.append(y.long())
        self.VAL_PRED.append(logits)

        return loss
    
    def on_validation_epoch_end(self):
        actual = torch.concat(self.VAL_ACTUAL, dim=0)
        preds = torch.concat(self.VAL_PRED, dim=0)
        
        acc = self.accuracy(preds, actual)
        mcc = self.mcc(preds, actual)

        self.log_dict({
            "val_acc": acc,
            "val_mcc": mcc
        }, prog_bar=True, on_epoch=True, on_step=False)

        self.VAL_ACTUAL.clear()
        self.VAL_PRED.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        label_loss = []
        label_columns = self.config.LABEL_COLUMNS[:-1]
        for i, label in enumerate(label_columns):
            pos_weight = self.config.get_pos_weight(i)
            loss = F.binary_cross_entropy_with_logits(logits[:, i], y[:, i], pos_weight=pos_weight)
            label_loss.append(loss)
            self.log(f"test_{label}_loss", loss)

        binary_loss = F.binary_cross_entropy_with_logits(logits[:, -1], y[:, -1], pos_weight=self.config.get_pos_weight(-1))
        self.log("test_binary_loss", binary_loss)

        loss = binary_loss * self.config.BINARY_WEIGHT + torch.mean(torch.tensor(label_loss)) * (1 - self.config.BINARY_WEIGHT)
        self.log("test_loss", loss)

        self.TEST_ACTUAL.append(y.long())
        self.TEST_PRED.append(logits)

        return loss
    
    def on_test_epoch_end(self):
        actual = torch.concat(self.TEST_ACTUAL, dim=0)
        preds = torch.concat(self.TEST_PRED, dim=0)
        
        acc = self.accuracy(preds, actual)
        mcc = self.mcc(preds, actual)

        self.log_dict({
            "test_acc": acc,
            "test_mcc": mcc
        }, prog_bar=True, on_epoch=True, on_step=False)

        self.TEST_ACTUAL.clear()
        self.TEST_PRED.clear()
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            amsgrad=True
        )
        return optimizer
    
    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     scheduler = optim.lr_scheduler.OneCycleLR(
    #         optimizer=optimizer,
    #         max_lr=self.learning_rate,
    #         epochs=self.config.N_EPOCHS,
    #         steps_per_epoch=self.config.N_STEPS_PER_EPOCH,
    #         pct_start=0.1,
    #         anneal_strategy="linear",
    #         cycle_momentum=True,
    #     )
    #     return (
    #         [optimizer],
    #         [scheduler]
    #     )
    

    

if __name__ == "__main__":
    model = Wav2Vec2Model(Config)
    model.training_step((torch.zeros(32, 16000), torch.zeros(32, 5)), None)
    model.on_train_epoch_end()
