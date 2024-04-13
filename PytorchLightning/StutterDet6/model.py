import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy, BinaryMatthewsCorrCoef
from torchmetrics.classification import MultilabelAccuracy, MultilabelMatthewsCorrCoef
import pytorch_lightning as pl
from transformers import AutoModelForAudioClassification

from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input_data):
        return input_data


class Wav2Vec2Model(pl.LightningModule):
    def __init__(
            self,
            config=Config
    ):
        super(Wav2Vec2Model, self).__init__()
        self.config = config
        self.n_outputs = self.config.N_OUTPUT_CLASS
        self.learning_rate = self.config.LEARNING_RATE

        self.w2v2_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=self.n_outputs)
        layer_to_unfreeze = self.config.LAYER_TO_UNFREEZE_FROM


        
        for param in self.w2v2_model.parameters():
            param.requires_grad = True
        for name, param in self.w2v2_model.named_parameters():
            if name in ['wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0', 'encoder.pos_conv_embed.conv.parametrizations.weight.original1']:
                param.requires_grad = True
            elif name == f'wav2vec2.encoder.layers.{layer_to_unfreeze}':
                break
            else:
                param.requires_grad = False
        self.w2v2_model.projector = Identity()
        self.w2v2_model.classifier = Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.n_outputs, bias=True)
        )

        if self.n_outputs > 1:
            self.accuracy = MultilabelAccuracy(num_labels=self.n_outputs)
            self.mcc = MultilabelMatthewsCorrCoef(num_labels=self.n_outputs)
        else:
            self.accuracy = BinaryAccuracy()
            self.mcc = BinaryMatthewsCorrCoef()

        self.pos_weight = 
            
        self.TRAIN_ACTUAL = []
        self.TRAIN_PRED = []
        self.VAL_ACTUAL = []
        self.VAL_PRED = []
        self.TEST_ACTUAL = []
        self.TEST_PRED = []


    def forward(self, input_data):
        out = self.w2v2_model(input_data).logits
        out = self.classifier(out)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits[:, -1], y[:, -1])

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

        loss = F.binary_cross_entropy_with_logits(logits[:, -1], y[:, -1])
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
    model = Wav2Vec2Model(Config).to(device)
    print(model.training_step((torch.zeros(32, 16000).to(device), torch.zeros(32, 5).to(device)), None))
    model.on_train_epoch_end()
