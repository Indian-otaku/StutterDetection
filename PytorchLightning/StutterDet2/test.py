from dataset import StutterDataModule, StutterDataset
import librosa
import torch
import matplotlib.pyplot as plt
import pandas as pd

# if 'IPython' not in globals():
#     dm = StutterDataModule(kfold_no=1)
#     dm.prepare_data()
#     dm.setup()
#     for x, y in dm.train_dataloader():
#         if y.sum() > 0:
#             print(x.shape, y.shape)
#             print(x.min(), x.max())
#             break
# else:
#     print("IPython in global")

ds = StutterDataset(df_path="E:\Desktop\StutterDetModel\data\KFold_dataset.csv")
x, y = ds[108]
print(x.shape, y.shape)
print(x.min(), x.max())
librosa.display.specshow(x.numpy())
plt.title(y.numpy())
plt.show()

# path = r"E:\Desktop\StutterDetModel\saved_models\multi_label-epoch=16-val_loss=0.998-2024-03-21_19-59-41.ckpt"
# model = StutterNetModel()

# print(torchaudio.list_audio_backends())


# accuracy = Accuracy(task="binary", num_classes=2)
# mcc = MatthewsCorrCoef(task="binary", num_classes=2)
# f1score = F1Score(task="binary", num_classes=2)

# checkpoint = torch.load(path)
# model.load_state_dict(state_dict=checkpoint['state_dict'])
# dm = Sep28kDataModule()
# # dm.prepare_data()
# dm.setup("predict")
# for x, y in dm.predict_dataloader():
#     if y.sum() > 0:
#         print(x.shape)
#         pred = model(x)
#         print(pred.shape, y.shape)
#         print(torch.sigmoid(pred).squeeze(), y.squeeze())
#         print(torch.round(torch.sigmoid(pred)).squeeze())
#         print(accuracy(pred, y))
#         print(mcc(pred, y))
#         print(f1score(pred, y))
#         break

# accuracy = Accuracy(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
# mcc = MatthewsCorrCoef(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
# f1score = F1Score(task="multilabel", num_labels=Config.N_OUTPUT_CLASS)
# cm = classification.MultilabelConfusionMatrix(num_labels=Config.N_OUTPUT_CLASS)

# checkpoint = torch.load(path)
# model.load_state_dict(state_dict=checkpoint['state_dict'])
# dm = StutterDataModule(kfold_no=1)
# dm.prepare_data()
# dm.setup()
# for x, y in dm.train_dataloader():
#     if y.sum() > 0:
#         print(x.shape, y.shape)
#         print(x.min(), x.max())
#         # pred = model(x)
#         # print(pred.shape, y.shape)
#         # print(torch.sigmoid(pred), y)
#         # print(torch.round(torch.sigmoid(pred)))
#         # print(accuracy(pred, y.long()))
#         # print(mcc(pred, y.long()))
#         # print(f1score(pred, y.long()))
#         # print(cm(pred, y.long()))
#         break


