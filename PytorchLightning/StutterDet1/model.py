import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchmetrics import F1Score, Accuracy, MatthewsCorrCoef
import pytorch_lightning as pl

from config import Config


class StatisticalPooling(nn.Module):
    def __init__(self):
        super(StatisticalPooling, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)
        return torch.cat((mean, std), dim=-1)
    
    
class StutterNetModel(pl.LightningModule):
    def __init__(
            self,
            dropout_rate=Config.DROPOUT_RATE,
            hidden_layers=Config.HIDDEN_LAYERS,
            learning_rate=Config.LEARNING_RATE
    ):
        super(StutterNetModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.N_MFCC, 
                out_channels=hidden_layers[0],
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[0])
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layers[0], 
                out_channels=hidden_layers[1],
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[1])
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layers[1], 
                out_channels=hidden_layers[2],
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[2])
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layers[2], 
                out_channels=hidden_layers[3],
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[3])
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layers[3], 
                out_channels=hidden_layers[4],
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[4])
        )
        self.statspool = StatisticalPooling()
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=hidden_layers[4]*2,
                out_features=hidden_layers[5],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[5]),
            nn.Dropout(p=dropout_rate),
            nn.Linear(
                in_features=hidden_layers[5],
                out_features=hidden_layers[6],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[6]),
            nn.Dropout(p=dropout_rate),
            nn.Linear(
                in_features=hidden_layers[6],
                out_features=hidden_layers[7],
            )
        )
        pos_weight = Config.get_pos_weight()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='mean'
        )
        self.learning_rate = learning_rate
        if Config.N_OUTPUT_CLASS > 1:
            self.accuracy = Accuracy(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
            self.mcc = MatthewsCorrCoef(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
            self.f1score = F1Score(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
        else:
            self.accuracy = Accuracy(task="binary", num_classes=2)
            self.mcc = MatthewsCorrCoef(task="binary", num_classes=2)
            self.f1score = F1Score(task="binary", num_classes=2)


    def forward(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.statspool(out)
        out = self.fc(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = self.accuracy(pred, y.long())
        mcc = self.mcc(pred, y.long())
        f1score = self.f1score(pred, y.long())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_mcc", mcc.float(), prog_bar=True, on_epoch=True)
        self.log("train_f1score", f1score, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = self.accuracy(pred, y.long())
        mcc = self.mcc(pred, y.long())
        f1score = self.f1score(pred, y.long())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_mcc", mcc.float(), prog_bar=True, on_epoch=True)
        self.log("val_f1score", f1score, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, y)
        acc = self.accuracy(pred, y).long()
        mcc = self.mcc(pred, y.long())
        f1score = self.f1score(pred, y.long())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_mcc", mcc.float(), prog_bar=True, on_epoch=True)
        self.log("test_f1score", f1score, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            amsgrad=True
        )
        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]