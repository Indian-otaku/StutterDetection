import os
import json
import random
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from config import Config



class StutterDataset(Dataset):
    
    def __init__(
            self, 
            data_dir=Config.DATA_DIR,
            stage="train", 
            label_columns=Config.LABEL_COLUMNS, 
            sample_rate=Config.SAMPLE_RATE,
    ):
        super(StutterDataset, self).__init__()
        self.audio_path = os.path.join(data_dir, r"clips/stuttering-clips/clips")
        self.df_path = os.path.join(data_dir, Config.DS_NAME)
        self.label_columns = label_columns
        self.sample_rate = sample_rate
        self.num_samples = self.sample_rate * 3
        _df = pd.read_csv(self.df_path)
        _df = _df[_df["SEP28k-E"] == stage]
        self.df = _df.reset_index(drop=True)
        self.noise_path = os.path.join(Config.NOISE_DATA_DIR, "fold1")
        self.noise_list = os.listdir(self.noise_path)
        self.stage = stage

    def __getitem__(self, idx):
        file_path = os.path.join(self.audio_path, str(self.df.loc[idx, "FileName"]))
        labels = np.array(self.df.loc[idx, self.label_columns].to_numpy().astype(np.float32))

        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate, mono=True, duration=3)
        except RuntimeError:
            print("RuntimeError:", file_path)
            signal = np.zeros((self.num_samples))
            sr = self.sample_rate
            labels = np.ones([len(self.label_columns)])

        signal = signal.astype(np.float32)
        signal = self._pad_if_necessary(signal)
        if self.stage == "train":
            signal = self._randomly_add_noise(signal)
        mfcc = librosa.feature.mfcc(
            y=signal, 
            
            sr=self.sample_rate, 
            n_mfcc=Config.N_MFCC, 
            n_mels=Config.N_MELS, 
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        mfcc = self._mean_normalization(mfcc)
        return mfcc, labels

    def _pad_if_necessary(self, signal):
        if signal.shape[-1] < self.num_samples:
            pad_len = self.num_samples - signal.shape[-1]
            signal = np.pad(signal, (0, pad_len))

        return signal
    
    def _randomly_add_noise(self, signal):
        i = random.randint(0, 5)
        if i > 0:
            noise_path = os.path.join(self.noise_path, random.choice(self.noise_list))
            noise, _ = librosa.load(noise_path, sr=self.sample_rate, mono=True, duration=3)
            noise = self._pad_if_necessary(noise)
            noise_normalized = noise * np.std(signal) / np.std(noise)
            return signal + 0.05 * noise_normalized
        return signal

    
    def _mean_normalization(self, mfcc):
        rows, cols = mfcc.shape
        
        mean = np.mean(mfcc, axis=-1)[:, np.newaxis]
        mean_vec = np.tile(mean, (1, cols))

        normalized_mfccs = mfcc - mean_vec
        return normalized_mfccs
    
    def __len__(self):
        return len(self.df)
    


class StutterDataModule(pl.LightningDataModule):
    
    def __init__(
            self, 
            data_dir=Config.DATA_DIR, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=Config.PIN_MEMORY,
            batch_size=Config.BATCH_SIZE, 
            label_columns=Config.LABEL_COLUMNS, 
            sample_rate=Config.SAMPLE_RATE, 
            *args, 
            **kwargs
    ):
        super(StutterDataModule, self).__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.label_columns = label_columns
        self.sample_rate = sample_rate
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

        if not os.path.exists(os.path.join(Config.NOISE_DATA_DIR, r"fold1")):
            print("Noise Data is downloaded from kaggle")
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "chrisfilo/urbansound8k",
                path=Config.NOISE_DATA_DIR,
                unzip=True
            )
            for i in range(1, 11):
                shutil.rmtree(os.path.join(Config.NOISE_DATA_DIR, f"fold{i}"))
            print("Noise Data was downloaded!")
        else:
            print("Noise Data is already available")

    def setup(self, stage : str):
        if stage == "fit":
            self.train = StutterDataset(
                data_dir=self.data_dir,
                stage="train",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
            self.val = StutterDataset(
                data_dir=self.data_dir,
                stage="dev",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
        if stage == "test":
            self.test = StutterDataset(
                data_dir=self.data_dir,
                stage="test",
                label_columns=self.label_columns,
                sample_rate=self.sample_rate
            )
        
        if stage == "predict":
            self.predict = StutterDataset(
                data_dir=self.data_dir,
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
            num_workers=self.num_workers
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

if __name__ == "__main__":
    ds = StutterDataset()
    print(ds[1][0].shape, ds[1][1])

    dm = StutterDataModule()
    dm.prepare_data()
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print(x.shape, y.shape)
        print(x.min(), x.max())
        print(x.mean(), x.std())
        break