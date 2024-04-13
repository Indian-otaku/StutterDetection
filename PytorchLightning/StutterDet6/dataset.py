import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio 
from torchaudio import transforms, functional
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from config import Config


class StutterDataset(Dataset):
    
    def __init__(
            self, 
            df_path,
            train=1,
            kfold_column="KFold1",
            config=Config,
    ):
        super(StutterDataset, self).__init__()
        self.config = config
        self.audio_path = os.path.join(config.DATA_DIR, r"clips/stuttering-clips/clips")
        self.label_columns = config.LABEL_COLUMNS
        self.sample_rate = config.SAMPLE_RATE
        self.num_samples = self.sample_rate * 3
        df = pd.read_csv(df_path)
        df = df[df[kfold_column] == int(train)]
        self.df = df.reset_index(drop=True)


    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_path, self.df.loc[idx, "FileName"])
        labels = torch.tensor(self.df.loc[idx, self.label_columns].to_numpy().astype(np.float32))

        try:
            signal, sr = torchaudio.load(file_path, format="wav")
        except RuntimeError:
            print("RuntimeError:", file_path)
            signal = torch.zeros((1, self.num_samples))
            sr = self.sample_rate
            labels = torch.zeros([len(self.label_columns)])

        labels = labels
        signal = signal
        
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_channels_if_necessary(signal)
        signal = self._trim_excess_if_necessary(signal)
        signal = self._pad_if_necessary(signal)
        signal = torch.squeeze(signal)

        return signal, labels


    def __len__(self):
        return len(self.df)


    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            signal = functional.resample(signal, orig_freq=sr, new_freq=self.sample_rate)
        return signal


    def _mix_channels_if_necessary(self, signal):
        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def _trim_excess_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal


    def _pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            pad_len = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, pad_len))
        return signal
    

class StutterDataModule(pl.LightningDataModule):
    
    def __init__(
            self, 
            kfold_no=1,
            config=Config
    ):
        super(StutterDataModule, self).__init__()
        self.config = config
        self.kfold_column = f"KFold{kfold_no}"
        self.df_path = os.path.join(self.config.DATA_DIR, self.config.DF_NAME)

    def prepare_data(self):
        with open(self.config.KAGGLE_KEY_PATH, "r") as f:
            data = json.load(f)
        os.environ["KAGGLE_USERNAME"] = data["username"]
        os.environ["KAGGLE_KEY"] = data["key"]
        del data

        if not os.path.exists(os.path.join(self.config.DATA_DIR, r"clips\stuttering-clips\clips")):
            print("Data is downloaded from kaggle")
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "bschuss02/sep28k",
                path=self.config.DATA_DIR,
                unzip=True
            )
            print("Data was downloaded!")

            os.remove(os.path.join(Config.DATA_DIR, "SEP-28k_episodes.csv"))
            os.remove(os.path.join(Config.DATA_DIR, "SEP-28k_labels.csv"))
        else:
            print("Data is already available")

    def setup(self, stage=None):
        self.train = StutterDataset(
            df_path=self.df_path,
            train=1,
            kfold_column=self.kfold_column
        )
        self.val = StutterDataset(
            df_path=self.df_path,
            train=0,
            kfold_column=self.kfold_column
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=self.config.PIN_MEMORY
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=self.config.PIN_MEMORY
        )
    
if __name__ == "__main__":
    print("Testing Dataset")
    ds = StutterDataset(r"E:\Desktop\StutterDetModel\data\KFold_dataset.csv")
    x, y = ds[1000]
    print(x.shape, y.shape)
    print(x.min(), x.max())

    print("Testing Datamodule")
    dl = StutterDataModule()
    dl.prepare_data()
    dl.setup("fit")
    for x, y in dl.train_dataloader():
        print(x.shape, y.shape)
        print(x.min(), x.max())
        print(x.mean(), x.std())
        break