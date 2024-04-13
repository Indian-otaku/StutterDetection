import os
import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from config import Config

class Sep28kDataset(Dataset):
    
    def __init__(
            self, 
            data_dir=Config.DATA_DIR, 
            data_split_column=Config.DATA_SPLIT_COLUMN, 
            stage="train", 
            label_columns=Config.LABEL_COLUMNS, 
            sample_rate=Config.SAMPLE_RATE
    ):
        self.audio_path = os.path.join(data_dir, r"clips/stuttering-clips/clips")
        self.df_path = os.path.join(data_dir, Config.DS_NAME)
        self.label_columns = label_columns
        self.sample_rate = sample_rate
        self.num_samples = self.sample_rate * 3
        _df = pd.read_csv(self.df_path)
        _df = _df[_df[data_split_column] == stage]
        self.df = _df.reset_index(drop=True)


    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_path, self.df.loc[idx, "FileName"])
        labels = self.df.loc[idx, self.label_columns].to_numpy().astype(np.float32)

        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True, duration=3)
        except RuntimeError:
            print("RuntimeError:", file_path)
            signal = np.zeros((1, self.num_samples))
            labels = np.zeros([len(self.label_columns)])

        signal = self._pad_if_necessary(signal)
        mfcc = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=Config.N_MFCC, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS, n_fft=Config.N_FFT)
        mfcc = self._min_max_normalize(mfcc)

        mfcc = torch.tensor(mfcc)
        labels = torch.tensor(labels)
        
        return mfcc, labels


    def __len__(self):
        return len(self.df)
    

    def _pad_if_necessary(self, signal):
        if signal.shape[-1] < self.num_samples:
            pad_len = self.num_samples - signal.shape[-1]
            signal = np.pad(signal, (0, pad_len))

        return signal
    

    def _cmvn(self, mfcc):
        eps = 2**-30
        rows, cols = mfcc.shape
        
        mean = np.mean(mfcc, axis=-1)[:, np.newaxis]
        mean_vec = np.tile(mean, (1, cols))

        normalized_mfccs = mfcc - mean_vec

        std = np.std(mfcc, axis=-1)[:, np.newaxis]
        std_vec = np.tile(std, (1, cols))

        normalized_mfccs = normalized_mfccs / (std_vec + eps)
        
        return normalized_mfccs
    
    
    def _min_max_normalize(self, mfcc):
        mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
        mfcc = mfcc * 2 - 1
        return mfcc
    


class Sep28kDataModule(pl.LightningDataModule):
    
    def __init__(
            self, 
            data_dir=Config.DATA_DIR, 
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
        else:
            print("Data is already available")

    def setup(self, stage : str):
        if stage == "fit":
            self.train = Sep28kDataset(
                data_dir=self.data_dir,
                data_split_column=self.data_split_column,
                stage="train",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
            self.val = Sep28kDataset(
                data_dir=self.data_dir,
                data_split_column=self.data_split_column,
                stage="dev",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
        if stage == "test":
            self.test = Sep28kDataset(
                data_dir=self.data_dir,
                data_split_column=self.data_split_column,
                stage="test",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
        
        if stage == "predict":
            self.predict = Sep28kDataset(
                data_dir=self.data_dir,
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