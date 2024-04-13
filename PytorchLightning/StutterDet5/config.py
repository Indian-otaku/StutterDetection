import torch
import pathlib

class Config:
    SAMPLE_RATE = 16000
    BINARY_WEIGHT = 0.5

    DATA_DIR = pathlib.Path("data")
    W2V2_SAVED_DIR = pathlib.Path("PytorchLightning/StutterDet5/w2v2data.pkl")
    DS_NAME = pathlib.Path(r"SEP28k_balanced.csv")
    LABEL_COLUMNS = ['Prolongation', 'Block', 'Interjection', 'Repetition', 'Stutter']
    N_OUTPUT_CLASS = len(LABEL_COLUMNS)
    KAGGLE_KEY_PATH = pathlib.Path("kaggle.json")
    LOG_DIR = pathlib.Path("PytorchLightning/StutterDet4/log")
    SAVED_MODEL_DIR = pathlib.Path("saved_models")

    NUM_WORKERS = 5
    PIN_MEMORY = True

    TRAIN_SPLIT = 0.8
    POS_WEIGHT_DICT = {
        'Prolongation': 7.597,
        'Block': 6.431,
        'Interjection': 3.314,
        'Repetition': 3.843,
        'Stutter': 1.0
    }

    N_EPOCHS = 15
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 256


    @staticmethod
    def get_pos_weight(index):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor([Config.POS_WEIGHT_DICT[Config.LABEL_COLUMNS[index]]], device=device)