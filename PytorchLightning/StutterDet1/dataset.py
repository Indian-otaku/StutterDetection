import os
import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio 
from torchaudio import transforms
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from config import Config

audio_transforms = transforms.MFCC(
    sample_rate=Config.SAMPLE_RATE,
    n_mfcc=Config.N_MFCC,
    melkwargs={"n_fft": Config.N_FFT, "hop_length": Config.HOP_LENGTH, "n_mels": Config.N_MELS}
)

class Sep28kDataset(Dataset):
    
    def __init__(
            self, 
            data_dir=Config.DATA_DIR, 
            transforms=audio_transforms, 
            data_split_column=Config.DATA_SPLIT_COLUMN, 
            stage="train", 
            label_columns=Config.LABEL_COLUMNS, 
            sample_rate=Config.SAMPLE_RATE
    ):
        self.audio_path = os.path.join(data_dir, r"clips/stuttering-clips/clips")
        self.df_path = os.path.join(data_dir, "SEP28K.csv")
        self.label_columns = label_columns
        self.sample_rate = sample_rate
        self.num_samples = self.sample_rate * 3
        self.transforms = transforms
        _df = pd.read_csv(self.df_path)
        _df = _df[_df[data_split_column] == stage]
        self.df = _df.reset_index(drop=True)


    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_path, self.df.loc[idx, "FileName"])
        labels = torch.tensor(self.df.loc[idx, self.label_columns].to_numpy().astype(np.float32))

        try:
            signal, sr = torchaudio.load(file_path, format="wav", normalize=True)
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
        signal = self.transforms(signal)
        signal = signal.squeeze()
        
        return signal, labels


    def __len__(self):
        return len(self.df)


    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
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
    

class Sep28kDataModule(pl.LightningDataModule):
    
    def __init__(
            self, 
            data_dir=Config.DATA_DIR, 
            transforms=audio_transforms, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=Config.PIN_MEMORY,
            batch_size=Config.BATCH_SIZE, 
            train_split=Config.TRAIN_SPLIT, 
            data_split_column=Config.DATA_SPLIT_COLUMN, 
            label_columns=Config.LABEL_COLUMNS, 
            sample_rate=Config.SAMPLE_RATE, 
            *args, 
            **kwargs
    ):
        super(Sep28kDataModule, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.data_split_column = data_split_column
        self.label_columns = label_columns
        self.sample_rate = sample_rate
        self.train_split = train_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        with open(Config.KAGGLE_KEY_PATH, "r") as f:
            data = json.load(f)
        os.environ["KAGGLE_USERNAME"] = data["username"]
        os.environ["KAGGLE_KEY"] = data["key"]
        del data

        if not os.path.exists(os.path.join(Config.DATA_DIR, r"clips\stuttering-clips\clips")):
            print("Data is downloaded from kaggle")
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "bschuss02/sep28k",
                path=Config.DATA_DIR,
                unzip=True
            )
            print("Data was downloaded!")

            os.remove(os.path.join(Config.DATA_DIR, "SEP-28k_episodes.csv"))
            os.remove(os.path.join(Config.DATA_DIR, "SEP-28k_labels.csv"))
            # os.remove(os.path.join(dataset_path, "sep28k.zip"))
        else:
            print("Data is already available")

    def setup(self, stage : str):
        if stage == "fit":
            self.train = Sep28kDataset(
                data_dir=self.data_dir,
                transforms=self.transforms,
                data_split_column=self.data_split_column,
                stage="train",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
            self.val = Sep28kDataset(
                data_dir=self.data_dir,
                transforms=self.transforms,
                data_split_column=self.data_split_column,
                stage="dev",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
        if stage == "test":
            self.test = Sep28kDataset(
                data_dir=self.data_dir,
                transforms=self.transforms,
                data_split_column=self.data_split_column,
                stage="test",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
        
        if stage == "predict":
            self.predict = Sep28kDataset(
                data_dir=self.data_dir,
                transforms=self.transforms,
                data_split_column=self.data_split_column,
                stage="test",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict,
            batch_size=self.batch_size,
            shuffle=False,
        )