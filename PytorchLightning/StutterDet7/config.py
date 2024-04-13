import torch

class Config:
    SAMPLE_RATE = 16000
    N_MFCC = 20
    N_MELS = 30
    FRAME_LENGTH_DURATION_IN_MS = 25
    FRAME_SKIP_DURATION_IN_MS = 12
    DATA_DIR = r"data"
    DS_NAME = r"SEP-28k_balanced_nostutter.csv"
    LABEL_COLUMNS = ['NoStutter']
    DATA_SPLIT_COLUMN = "SEP28k-E"
    N_OUTPUT_CLASS = len(LABEL_COLUMNS)
    DROPOUT_RATE = 0.3
    HIDDEN_LAYERS = [
        64, 
        64, 
        64, 
        64, 
        3*64,
        64,
        3*2*64,
        64,
        64,
        N_OUTPUT_CLASS
    ]
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 5
    PIN_MEMORY = True
    BATCH_SIZE = 128
    N_EPOCHS = 50
    KAGGLE_KEY_PATH = r"C://Users//Asus//Downloads//kaggle.json"
    LOG_DIR = r"E://Desktop//StutterDetModel//PytorchLightning//StutterDet1//log"
    SAVED_MODEL_DIR = r"E://Desktop//StutterDetModel//saved_models"