from model import StutterNetModel
from dataset import Sep28kDataModule
import torch
import torchaudio
from torchmetrics import Accuracy, MatthewsCorrCoef, F1Score, classification
from config import Config
import pytorch_lightning as pl
import matplotlib.pyplot as plt

torch.random.manual_seed(40)

path = r"E:\Desktop\StutterDetModel\saved_models\multi_label-epoch=09-val_loss=0.758-2024-03-26_20-25-18.ckpt"
model = StutterNetModel()

# print(torchaudio.list_audio_backends())


accuracy = Accuracy(task="binary", num_classes=2)
mcc = MatthewsCorrCoef(task="binary", num_classes=2)
f1score = F1Score(task="binary", num_classes=2)

checkpoint = torch.load(path)
# model.load_state_dict(state_dict=checkpoint['state_dict'])
dm = Sep28kDataModule()
# dm.prepare_data()
dm.setup("fit")
for x, y in dm.train_dataloader():
    if y.sum() > 0:
        print(x.min(), x.max())
        print(torch.mean(x), torch.std(x))
#         print(x.shape)
#         pred = model(x)
#         plt.imshow(X=x[0], vmax=5, vmin=-100)
#         plt.show()
#         print(pred.shape, y.shape)
#         print(torch.sigmoid(pred).squeeze(), y.squeeze())
#         print(torch.round(torch.sigmoid(pred)).squeeze())
#         print(accuracy(pred, y))
#         print(mcc(pred, y))
#         print(f1score(pred, y))
        break

# # accuracy = Accuracy(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
# # mcc = MatthewsCorrCoef(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
# # f1score = F1Score(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
# # cm = classification.MultilabelConfusionMatrix(num_labels=Config.N_OUTPUT_CLASS)

# # checkpoint = torch.load(path)
# # model.load_state_dict(state_dict=checkpoint['state_dict'])
# # dm = Sep28kDataModule()
# # # dm.prepare_data()
# # dm.setup("predict")
# # for x, y in dm.predict_dataloader():
# #     if y.sum() > 0:
# #         print(x.shape)
# #         pred = model(x)
# #         print(pred.shape, y.shape)
# #         print(torch.sigmoid(pred), y)
# #         print(torch.round(torch.sigmoid(pred)))
# #         print(accuracy(pred, y.long()))
# #         print(mcc(pred, y.long()))
# #         print(f1score(pred, y.long()))
# #         print(cm(pred, y.long()))
# #         break


