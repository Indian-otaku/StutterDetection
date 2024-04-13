import os
import json
import warnings
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import pytorch_lightning as pl
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from config import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

class Wav2Vec2ExtractFeatures:
    def __init__(self, sample_rate):
        self.input_prep = None
        self.feature_extractor = None
        self.sample_rate = sample_rate

    def initialize(self):
        self.input_prep = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

    def extract_features(self, signal):
        if self.input_prep is None or self.feature_extractor is None:
            self.initialize()
        
        with torch.no_grad():
            wav2vec2_features = self.input_prep(signal, sampling_rate=self.sample_rate, return_tensors="pt")
            wav2vec2_features = wav2vec2_features.input_values[0].to(device)
            wav2vec2_features = torch.unsqueeze(wav2vec2_features, dim=0)
            wav2vec2_features = self.feature_extractor(wav2vec2_features)
            wav2vec2_features = torch.concat([wav2vec2_features.last_hidden_state, wav2vec2_features.extract_features], dim=2)
            wav2vec2_features = torch.squeeze(wav2vec2_features)
            wav2vec2_features = torch.mean(wav2vec2_features, dim=0)
            wav2vec2_features = wav2vec2_features.to("cpu").numpy()

        return wav2vec2_features

# class GetW2V2Features():
#     def __init__(
#             self, 
#             df,
#             config=Config,
#     ):
#         self.config = config
#         self.audio_path = os.path.join(config.DATA_DIR, r"clips/stuttering-clips/clips")
#         self.df_path = os.path.join(config.DATA_DIR, config.DS_NAME)
#         self.label_columns = config.LABEL_COLUMNS
#         self.sample_rate = config.SAMPLE_RATE
#         self.num_samples = self.sample_rate * 3
#         self.df = df
#         self.w2v2_extractor = Wav2Vec2ExtractFeatures(self.sample_rate)

#     def get_w2v2_features(self):
#         w2v2_out = []
#         for filename in self.df["FileName"]:
#             w2v2_out.append(self.get_single_w2v2_features(filename))
#         with open(self.config.W2V2_SAVED_DIR, "wb") as f:
#             pickle.dump(w2v2_out, f)
#         return w2v2_out
    
#     def get_single_w2v2_features(self, filename):
#         path = os.path.join(self.audio_path, filename)
#         try:
#             signal, _ = librosa.load(path, sr=self.sample_rate, mono=True, duration=3)
#         except RuntimeError:
#             print("RuntimeError:", path)
#             signal = np.zeros((1, self.num_samples))
#         signal = self._pad_if_necessary(signal)
#         return self.w2v2_extractor.extract_features(signal)
        
#     def _pad_if_necessary(self, signal):
#         if signal.shape[-1] < self.num_samples:
#             pad_len = self.num_samples - signal.shape[-1]
#             signal = np.pad(signal, (0, pad_len))
#         return signal



class GetW2V2Features():
    def __init__(
            self, 
            df,
            config=Config,
    ):
        self.config = config
        self.audio_path = os.path.join(config.DATA_DIR, r"clips/stuttering-clips/clips")
        self.df_path = os.path.join(config.DATA_DIR, config.DS_NAME)
        self.label_columns = config.LABEL_COLUMNS
        self.sample_rate = config.SAMPLE_RATE
        self.num_samples = self.sample_rate * 3
        self.df = df
        self.w2v2_extractor = Wav2Vec2ExtractFeatures(self.sample_rate)

    def get_w2v2_features(self):
        w2v2_out = []
        for filename in self.df["FileName"]:
            w2v2_out.append(self.get_single_w2v2_features(filename))
        with open(self.config.W2V2_SAVED_DIR, "wb") as f:
            pickle.dump(w2v2_out, f)
        return w2v2_out
    
    def get_single_w2v2_features(self, filename):
        path = os.path.join(self.audio_path, filename)
        try:
            signal, _ = librosa.load(path, sr=self.sample_rate, mono=True, duration=3)
        except RuntimeError:
            print("RuntimeError:", path)
            signal = np.zeros((1, self.num_samples))
        signal = self._pad_if_necessary(signal)
        return self.w2v2_extractor.extract_features(signal)
        
    def _pad_if_necessary(self, signal):
        if signal.shape[-1] < self.num_samples:
            pad_len = self.num_samples - signal.shape[-1]
            signal = np.pad(signal, (0, pad_len))
        return signal



class StutterDataset(Dataset):
    def __init__(
            self, 
            data_dir=Config.DATA_DIR,
            stage="train", 
            stage_size=6000,
            label_columns=Config.LABEL_COLUMNS, 
            sample_rate=Config.SAMPLE_RATE
    ):
        super(StutterDataset, self).__init__()
        self.audio_path = os.path.join(data_dir, r"clips/stuttering-clips/clips")
        self.df_path = os.path.join(data_dir, Config.DS_NAME)
        self.label_columns = label_columns
        self.sample_rate = sample_rate
        self.num_samples = self.sample_rate * 3
        _df = pd.read_csv(self.df_path)
        _df = _df[_df["SEP28k-E"] == stage].sample(stage_size)
        self.df = _df.reset_index(drop=True)

        if os.path.exists("w2v2data.pkl"):
            with open(Config.W2V2_SAVED_DIR, "rb") as f:
                self.w2v2_features = pickle.load(f)
        else:
            self.w2v2_features = GetW2V2Features(self.df).get_w2v2_features()

    def __getitem__(self, idx):
        labels = self.df.loc[idx, self.label_columns].to_numpy().astype(np.float32)
        wav2vec2_features = self.w2v2_features[idx]
        return wav2vec2_features, labels

    def __len__(self):
        return len(self.df)

    


# class StutterDataModule(pl.LightningDataModule):
    
#     def __init__(
#             self, 
#             data_dir=Config.DATA_DIR, 
#             num_workers=Config.NUM_WORKERS, 
#             pin_memory=Config.PIN_MEMORY,
#             batch_size=Config.BATCH_SIZE, 
#             train_split=Config.TRAIN_SPLIT,
#             label_columns=Config.LABEL_COLUMNS, 
#             sample_rate=Config.SAMPLE_RATE, 
#             *args, 
#             **kwargs
#     ):
#         super(StutterDataModule, self).__init__(*args, **kwargs)
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.label_columns = label_columns
#         self.sample_rate = sample_rate
#         self.train_split = train_split
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory

#     def prepare_data(self):
#         with open(Config.KAGGLE_KEY_PATH, "r") as f:
#             data = json.load(f)
#         os.environ["KAGGLE_USERNAME"] = data["username"]
#         os.environ["KAGGLE_KEY"] = data["key"]
#         del data

#         if not os.path.exists(os.path.join(Config.DATA_DIR, r"clips\stuttering-clips\clips")):
#             print("Data is downloaded from kaggle")
#             import kaggle
#             kaggle.api.authenticate()
#             kaggle.api.dataset_download_files(
#                 "bschuss02/sep28k",
#                 path=Config.DATA_DIR,
#                 unzip=True
#             )
#             print("Data was downloaded!")

#             os.remove(os.path.join(Config.DATA_DIR, "SEP-28k_episodes.csv"))
#             os.remove(os.path.join(Config.DATA_DIR, "SEP-28k_labels.csv"))
#         else:
#             print("Data is already available")

#     def setup(self, stage : str):
#         if stage == "fit":
#             self.train = StutterDataset(
#                 data_dir=self.data_dir,
#                 stage="train",
#                 label_columns=self.label_columns,
#                 sample_rate=self.sample_rate
#             )
#             self.val = StutterDataset(
#                 data_dir=self.data_dir,
#                 stage="dev",
#                 label_columns=self.label_columns,
#                 sample_rate=self.sample_rate
#             )
#         if stage == "test":
#             self.test = StutterDataset(
#                 data_dir=self.data_dir,
#                 stage="test",
#                 label_columns=self.label_columns,
#                 sample_rate=self.sample_rate
#             )
        
#         if stage == "predict":
#             self.predict = StutterDataset(
#                 data_dir=self.data_dir,
#                 stage="test",
#                 label_columns=self.label_columns,
#                 sample_rate=self.sample_rate
#             )
    
#     def train_dataloader(self):
#         return DataLoader(
#             dataset=self.train,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             persistent_workers=True,
#             pin_memory=self.pin_memory
#         )
    
#     def val_dataloader(self):
#         return DataLoader(
#             dataset=self.val,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             persistent_workers=True,
#             pin_memory=self.pin_memory
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             dataset=self.test,
#             batch_size=self.batch_size,
#             shuffle=False,
#         )
    
#     def predict_dataloader(self):
#         return DataLoader(
#             dataset=self.predict,
#             batch_size=self.batch_size,
#             shuffle=False,
#         )
    

if __name__ == "__main__":
    w2v2_extractor = Wav2Vec2ExtractFeatures(Config.SAMPLE_RATE)
    print(w2v2_extractor.extract_features(librosa.load(r"StutterDetModel/data/clips/stuttering-clips/clips/FluencyBank_010_2.wav")[0]).shape)

    ds = StutterDataset()
    print(ds[0])

    # ds = StutterDataset()
    # x, y = ds[0]
    # print(x.shape, y.shape)
    
    # dl = StutterDataModule()
    # dl.prepare_data()
    # dl.setup("fit")
    # for x, y in dl.train_dataloader():
    #     print(x.shape, y.shape)
    #     break