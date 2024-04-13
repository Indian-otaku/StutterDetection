import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import F1Score, Accuracy, MatthewsCorrCoef
from torchmetrics.functional import concordance_corrcoef
import pytorch_lightning as pl

from config import Config

class CCCLoss(torch.nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, preds, target):
        ccc = concordance_corrcoef(preds, target)
        return (1 - ccc)

class StatisticalPooling(nn.Module):
    def __init__(self):
        super(StatisticalPooling, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)
        return torch.cat((mean, std), dim=-1)
    
    
class StutterModel(pl.LightningModule):
    def __init__(
            self,
            gamma=2,
            config=Config,
    ):
        super(StutterModel, self).__init__()
        self.config = config
        hidden_layers = config.HIDDEN_LAYERS
        dropout_rate = config.DROPOUT_RATE
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.N_MFCC, 
                out_channels=hidden_layers[0],
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1
            ),
            nn.BatchNorm1d(num_features=hidden_layers[0]),
            nn.ReLU()
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
            nn.BatchNorm1d(num_features=hidden_layers[1]),
            nn.ReLU()
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
            nn.BatchNorm1d(num_features=hidden_layers[2]),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layers[2], 
                out_channels=hidden_layers[3],
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=hidden_layers[3]),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_layers[3], 
                out_channels=hidden_layers[4],
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=hidden_layers[4]),
            nn.ReLU()
        )
        self.statspool = StatisticalPooling()
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=hidden_layers[4]*2,
                out_features=hidden_layers[5],
            ),
            nn.BatchNorm1d(num_features=hidden_layers[5]),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(
                in_features=hidden_layers[5],
                out_features=hidden_layers[6],
            ),
            nn.BatchNorm1d(num_features=hidden_layers[6]),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(
                in_features=hidden_layers[6],
                out_features=hidden_layers[7],
            )
        )
        self.binary_criterion = nn.BCEWithLogitsLoss()
        self.multitask_criterion = CCCLoss()
        self.learning_rate = config.LEARNING_RATE
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
        
        multitask_loss = []
        for i in range(len(self.config.LABEL_COLUMNS)-1):
            loss = self.binary_criterion(pred[:, i], y[:, i])
            multitask_loss.append(loss)
            self.log(f"train_{self.config.LABEL_COLUMNS[i]}_loss", loss)

        binary_loss = self.binary_criterion(logits[:, -1], y[:, -1])
        self.log("train_binary_loss", binary_loss, prog_bar=True)
        multitask_loss = torch.mean(torch.tensor(multitask_loss)) 
        self.log("train_multi_loss", multitask_loss, prog_bar=True)

        loss = (binary_loss * self.config.BINARY_WEIGHT) \
              + multitask_loss * (1 - self.config.BINARY_WEIGHT)
        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.softmax(logits, dim=1)
        
        multitask_loss = []
        for i in range(len(self.config.LABEL_COLUMNS)-1):
            loss = self.multitask_criterion(pred[:, i], y[:, i])
            multitask_loss.append(loss)
            self.log(f"val_{self.config.LABEL_COLUMNS[i]}_loss", loss)

        binary_loss = self.binary_criterion(logits[:, -1], y[:, -1])
        self.log("val_binary_loss", binary_loss)

        loss = (binary_loss * self.config.BINARY_WEIGHT) \
              + torch.sum(torch.tensor(multitask_loss)) * (1 - self.config.BINARY_WEIGHT)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
            amsgrad=True
        )
        return [optimizer]