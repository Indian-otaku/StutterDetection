import torch

class Config:
    SAMPLE_RATE = 16000
    N_MFCC = 20
    N_MELS = 30
    N_FFT = 512
    HOP_LENGTH = 256
    DATA_DIR = r"data"
    DF_NAME = r"KFold_dataset.csv"
    LABEL_COLUMNS = ['Prolongation', 'Repetition', 'Block', 'Interjection', 'NoStutter']
    DATA_SPLIT_COLUMN = "SEP28k-E"
    N_OUTPUT_CLASS = len(LABEL_COLUMNS)
    DROPOUT_RATE = 0.5
    HIDDEN_LAYERS = [
        256, 
        256, 
        256, 
        256, 
        512,
        256,
        256,
        N_OUTPUT_CLASS
    ]
    LEARNING_RATE = 0.005
    NUM_WORKERS = 1
    PIN_MEMORY = True
    BATCH_SIZE = 64
    TRAIN_SPLIT = 0.8
    KAGGLE_KEY_PATH = r"" # Give the path where kaggle.json api file is stored. 
    LOG_DIR = r"PytorchLightning\StutterDet1\log"
    SAVED_MODEL_DIR = r"saved_models"
    BINARY_WEIGHT = 0.5