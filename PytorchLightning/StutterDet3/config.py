import torch

class Config:
    SAMPLE_RATE = 16000
    N_MFCC = 20
    N_MELS = 30
    FRAME_LENGTH_DURATION_IN_MS = 25
    FRAME_SKIP_DURATION_IN_MS = 12
    N_FFT = 512
    HOP_LENGTH = N_FFT // 2
    DATA_DIR = r"data"
    DS_NAME = r"SEP28k_balanced.csv"
    LABEL_COLUMNS = ['Stutter']
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
    NUM_WORKERS = 5
    PIN_MEMORY = True
    BATCH_SIZE = 64
    TRAIN_SPLIT = 0.8
    KAGGLE_KEY_PATH = r"" # Give the path where kaggle.json api file is stored. 
    LOG_DIR = r"PytorchLightning\StutterDet1\log"
    SAVED_MODEL_DIR = r"saved_models"
    POS_WEIGHT_DICT = {
        'Prolongation': 7.597,
        'Block': 6.431,
        'Interjection': 3.314,
        'Repetition': 3.843,
        'Stutter': 1.0
    }

    @staticmethod
    def get_pos_weight():
        return torch.tensor([Config.POS_WEIGHT_DICT[col] for col in Config.LABEL_COLUMNS])