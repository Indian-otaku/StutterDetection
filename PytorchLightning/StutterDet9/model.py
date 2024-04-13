import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchmetrics.classification import BinaryAccuracy, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.classification import MultilabelAccuracy, MultilabelMatthewsCorrCoef, MultilabelF1Score
import pytorch_lightning as pl

from config import Config


class StatisticalPooling(nn.Module):
    def __init__(self):
        super(StatisticalPooling, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)
        return torch.cat((mean, std), dim=-1)
    
class StutterNetHead(nn.Module):
    def __init__(
            self,
            context_setting=(5, 2), # (9, 4)
            hidden_layers=Config.HIDDEN_LAYERS
    ):
        super(StutterNetHead, self).__init__()
        self.hidden_layers = hidden_layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.N_MFCC, 
                out_channels=hidden_layers[0],
                kernel_size=context_setting[0],
                stride=1,
                padding=context_setting[1],
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
        self.bilstm = nn.LSTM(
            input_size=301,
            hidden_size=hidden_layers[5],
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.statspool = StatisticalPooling()
    
    def forward(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        hidden = (
            torch.zeros(2*2, out.size(0), self.hidden_layers[5]).requires_grad_(True).to("cuda"),
            torch.zeros(2*2, out.size(0), self.hidden_layers[5]).requires_grad_(True).to("cuda")
        )
        out, _ = self.bilstm(out, hidden)
        out = self.statspool(out)
        return out
    

class StutterNetModel(pl.LightningModule):
    def __init__(
            self,
            dropout_rate=Config.DROPOUT_RATE,
            hidden_layers=Config.HIDDEN_LAYERS,
            learning_rate=Config.LEARNING_RATE
    ):
        super(StutterNetModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.stutternet_context5 = StutterNetHead(
            context_setting=(5, 2),
            hidden_layers=hidden_layers
        )
        self.stutternet_context9 = StutterNetHead(
            context_setting=(9, 4),
            hidden_layers=hidden_layers
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=hidden_layers[6],
                out_features=hidden_layers[7],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[7]),
            nn.Dropout(p=dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=hidden_layers[7],
                out_features=hidden_layers[8],
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_layers[8]),
            nn.Dropout(p=dropout_rate)
        )
        self.fc3 = nn.Linear(
            in_features=hidden_layers[8],
            out_features=hidden_layers[9],
        )
        
        self.criterion = nn.BCEWithLogitsLoss(
            reduction='mean'
        )
        self.learning_rate = learning_rate
        if Config.N_OUTPUT_CLASS > 1:
            self.accuracy = MultilabelAccuracy(num_labels=Config.N_OUTPUT_CLASS)
            self.mcc = MultilabelMatthewsCorrCoef(num_labels=Config.N_OUTPUT_CLASS)
            self.f1score = MultilabelF1Score(num_labels=Config.N_OUTPUT_CLASS)
        else:
            self.accuracy = BinaryAccuracy()
            self.mcc = BinaryMatthewsCorrCoef()
            self.f1score = BinaryF1Score()

        self.TRAIN_ACTUAL = []
        self.TRAIN_PRED = []
        self.VAL_ACTUAL = []
        self.VAL_PRED = []
        self.TEST_ACTUAL = []
        self.TEST_PRED = []


    def forward(self, input_data):
        out1 = self.stutternet_context5(input_data)
        out2 = self.stutternet_context9(input_data)
        out = torch.concatenate([out1, out2], dim=-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.sigmoid(logits)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.TRAIN_ACTUAL.append(y.long())
        self.TRAIN_PRED.append(pred)
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
        pred = torch.sigmoid(logits)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.VAL_ACTUAL.append(y.long())
        self.VAL_PRED.append(pred)
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
        pred = torch.sigmoid(logits)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.TEST_ACTUAL.append(y.long())
        self.TEST_PRED.append(pred)
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
            weight_decay=1e-5
        )
        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            ),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
    
    
# class StutterNetModel(pl.LightningModule):
#     def __init__(
#             self,
#             dropout_rate=Config.DROPOUT_RATE,
#             hidden_layers=Config.HIDDEN_LAYERS,
#             learning_rate=Config.LEARNING_RATE
#     ):
#         super(StutterNetModel, self).__init__()
#         self.hidden_layers = hidden_layers
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=Config.N_MFCC, 
#                 out_channels=hidden_layers[0],
#                 kernel_size=5,
#                 stride=1,
#                 padding=2,
#                 dilation=1
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[0])
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_layers[0], 
#                 out_channels=hidden_layers[1],
#                 kernel_size=3,
#                 stride=1,
#                 padding=2,
#                 dilation=2
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[1])
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_layers[1], 
#                 out_channels=hidden_layers[2],
#                 kernel_size=3,
#                 stride=1,
#                 padding=3,
#                 dilation=3
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[2])
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_layers[2], 
#                 out_channels=hidden_layers[3],
#                 kernel_size=1,
#                 stride=1,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[3])
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=hidden_layers[3], 
#                 out_channels=hidden_layers[4],
#                 kernel_size=1,
#                 stride=1,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[4])
#         )
#         self.bilstm = nn.LSTM(
#             input_size=301,
#             hidden_size=hidden_layers[5],
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )
#         self.bilstm_out = nn.Sequential(
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[4])
#         )
#         self.statspool = StatisticalPooling()
#         self.fc1 = nn.Sequential(
#             nn.Linear(
#                 in_features=hidden_layers[6],
#                 out_features=hidden_layers[7],
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[7]),
#             nn.Dropout(p=dropout_rate)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(
#                 in_features=hidden_layers[7],
#                 out_features=hidden_layers[8],
#             ),
#             nn.ReLU(),
#             nn.BatchNorm1d(num_features=hidden_layers[8]),
#             nn.Dropout(p=dropout_rate)
#         )
#         self.fc3 = nn.Linear(
#             in_features=hidden_layers[8],
#             out_features=hidden_layers[9],
#         )
        
#         self.criterion = nn.BCEWithLogitsLoss(
#             reduction='mean'
#         )
#         self.learning_rate = learning_rate
#         if Config.N_OUTPUT_CLASS > 1:
#             self.accuracy = MultilabelAccuracy(num_labels=Config.N_OUTPUT_CLASS)
#             self.mcc = MultilabelMatthewsCorrCoef(num_labels=Config.N_OUTPUT_CLASS)
#             self.f1score = MultilabelF1Score(num_labels=Config.N_OUTPUT_CLASS)
#         else:
#             self.accuracy = BinaryAccuracy()
#             self.mcc = BinaryMatthewsCorrCoef()
#             self.f1score = BinaryF1Score()

#         self.TRAIN_ACTUAL = []
#         self.TRAIN_PRED = []
#         self.VAL_ACTUAL = []
#         self.VAL_PRED = []
#         self.TEST_ACTUAL = []
#         self.TEST_PRED = []


#     def forward(self, input_data):
#         out = self.conv1(input_data)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#         hidden = (
#             torch.randn(2*2, out.size(0), self.hidden_layers[5]).requires_grad_(True).to("cuda"),
#             torch.randn(2*2, out.size(0), self.hidden_layers[5]).requires_grad_(True).to("cuda")
#         )
#         out, _ = self.bilstm(out, hidden)
#         out = self.bilstm_out(out)
#         out = self.statspool(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
#         return out
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         pred = torch.sigmoid(logits)
#         loss = self.criterion(logits, y)
#         self.log("train_loss", loss, prog_bar=True)
#         self.TRAIN_ACTUAL.append(y.long())
#         self.TRAIN_PRED.append(pred)
#         return loss
    
#     def on_train_epoch_end(self):
#         actual = torch.concat(self.TRAIN_ACTUAL, dim=0)
#         preds = torch.concat(self.TRAIN_PRED, dim=0)
        
#         acc = self.accuracy(preds, actual)
#         mcc = self.mcc(preds, actual)

#         self.log_dict({
#             "train_acc": acc,
#             "train_mcc": mcc
#         }, prog_bar=True, on_epoch=True, on_step=False)

#         self.TRAIN_ACTUAL.clear()
#         self.TRAIN_PRED.clear()
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         pred = torch.sigmoid(logits)
#         loss = self.criterion(logits, y)
#         self.log("val_loss", loss, prog_bar=True)
#         self.VAL_ACTUAL.append(y.long())
#         self.VAL_PRED.append(pred)
#         return loss
    
#     def on_validation_epoch_end(self):
#         actual = torch.concat(self.VAL_ACTUAL, dim=0)
#         preds = torch.concat(self.VAL_PRED, dim=0)
        
#         acc = self.accuracy(preds, actual)
#         mcc = self.mcc(preds, actual)

#         self.log_dict({
#             "val_acc": acc,
#             "val_mcc": mcc
#         }, prog_bar=True, on_epoch=True, on_step=False)

#         self.VAL_ACTUAL.clear()
#         self.VAL_PRED.clear()
    
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         pred = torch.sigmoid(logits)
#         loss = self.criterion(logits, y)
#         self.log("test_loss", loss, prog_bar=True)
#         self.TEST_ACTUAL.append(y.long())
#         self.TEST_PRED.append(pred)
#         return loss
    
#     def on_test_epoch_end(self):
#         actual = torch.concat(self.TEST_ACTUAL, dim=0)
#         preds = torch.concat(self.TEST_PRED, dim=0)
        
#         acc = self.accuracy(preds, actual)
#         mcc = self.mcc(preds, actual)

#         self.log_dict({
#             "test_acc": acc,
#             "test_mcc": mcc
#         }, prog_bar=True, on_epoch=True, on_step=False)

#         self.TEST_ACTUAL.clear()
#         self.TEST_PRED.clear()
    
#     def configure_optimizers(self):
#         optimizer = Adam(
#             self.parameters(),
#             lr=self.learning_rate,
#             weight_decay=1e-5
#         )
#         return optimizer
    

if __name__ == "__main__":
    model = StutterNetModel().to("cuda")
    print(model)
    print(model(torch.rand(Config.BATCH_SIZE, 20, 301).to("cuda")).shape)