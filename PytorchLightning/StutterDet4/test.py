from model import Wav2Vec2Model
from dataset import Sep28kDataModule, Sep28kDataset
import random
import torch
from torchmetrics.classification import MultilabelAccuracy, MultilabelMatthewsCorrCoef, MultilabelF1Score, MultilabelConfusionMatrix, BinaryF1Score
from config import Config
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action="ignore")


pl.seed_everything(100)

path = r"E:\Desktop\StutterDetModel\saved_models\w2v2-epoch=08-val_loss=0.786-2024-03-27_22-39-03.ckpt"
model = Wav2Vec2Model()

accuracy = MultilabelAccuracy(num_labels=Config.N_OUTPUT_CLASS)
mcc = MultilabelMatthewsCorrCoef(num_labels=Config.N_OUTPUT_CLASS)
f1score = MultilabelF1Score(num_labels=Config.N_OUTPUT_CLASS)
cm = MultilabelConfusionMatrix(num_labels=Config.N_OUTPUT_CLASS)

checkpoint = torch.load(path)
model.load_state_dict(state_dict=checkpoint['state_dict'])

dm = Sep28kDataModule()

print(len(Sep28kDataset()))
print(len(Sep28kDataset())/Config.BATCH_SIZE)

dm.prepare_data()
dm.setup("predict")


# for x, y in dm.predict_dataloader():
#     if y.sum() > 0:
#         pred = model(x)
#         print(x.shape, pred.shape, y.shape)
#         print("-"*100)
#         print(torch.sigmoid(pred), y, sep="\n")
#         print(torch.round(torch.sigmoid(pred)))
#         print("-"*100)
#         print("Accuracy:", accuracy(pred, y.long()).numpy())
#         print("MCC:", mcc(pred, y.long()).numpy())
#         print("F1Score:", f1score(pred, y.long()).numpy())
#         print("Confusion matrix:\n", cm(pred, y.long()).numpy())
#         break


