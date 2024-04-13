import torch

class Config:
    SAMPLE_RATE = 16000
    BINARY_WEIGHT = 0.5

    DATA_DIR = r"data"
    DF_NAME = r"KFold_dataset.csv"
    LABEL_COLUMNS = ['NoStutter']
    N_OUTPUT_CLASS = len(LABEL_COLUMNS)
    KAGGLE_KEY_PATH = r"" # Give the path where kaggle.json api file is stored. 
    LOG_DIR = r"PytorchLightning\StutterDet4\log"
    SAVED_MODEL_DIR = r"saved_models"

    NUM_WORKERS = 10
    PIN_MEMORY = False

    TRAIN_SPLIT = 0.8
    N_EPOCHS = 100
    LAYER_TO_UNFREEZE_FROM = 11
    LEARNING_RATE = 5e-3
    BATCH_SIZE = 32

    POS_WEIGHT_DICT = {
        "NoStutter"
    }


    @staticmethod
    def get_pos_weight(index):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor([Config.POS_WEIGHT_DICT[Config.LABEL_COLUMNS[index]]], device=device)