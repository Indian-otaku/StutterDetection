import torch

class Config:
    SAMPLE_RATE = 16000
    BINARY_WEIGHT = 0.5

    DATA_DIR = r"data"
    DS_NAME = r"SEP-28k_balanced_nostutter.csv"
    LABEL_COLUMNS = ['NoStutter']
    N_OUTPUT_CLASS = len(LABEL_COLUMNS)
    KAGGLE_KEY_PATH = r"" # Give the path where kaggle.json api file is stored. 
    LOG_DIR = r"PytorchLightning\StutterDet4\log"
    SAVED_MODEL_DIR = r"saved_models"

    NUM_WORKERS = 5
    PIN_MEMORY = False

    TRAIN_SPLIT = 0.8
    POS_WEIGHT_DICT = {
        'Prolongation': 7.591,
        'Block': 6.431,
        'Interjection': 3.325,
        'Repetition': 3.845,
        'NoStutter': 1.0
    }

    N_EPOCHS = 15
    LAYER_TO_UNFREEZE_FROM = 11
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 16


    @staticmethod
    def get_pos_weight(index):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor([Config.POS_WEIGHT_DICT[Config.LABEL_COLUMNS[index]]], device=device)