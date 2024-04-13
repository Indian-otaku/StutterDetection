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
        'SEP28k-T': {
            'Prolongation': 8.790839694656489,
            'Block': 7.393979057591623,
            'Interjection': 4.069565217391304,
            'Repetition': 4.993457943925233,
            'Stutter': 1.4054763690922731
        },
        'SEP28k-D': {
            'Prolongation': 8.853968253968254,
            'Block': 6.818639798488665,
            'Interjection': 4.243243243243243,
            'Repetition': 4.3517241379310345,
            'Stutter': 1.278165137614679
        },
        'SEP28k-E': {
            'Prolongation': 8.971679028995279,
            'Block': 7.678403755868545,
            'Interjection': 3.281412854661262,
            'Repetition': 4.678955453149001,
            'Stutter': 1.3465566486829579
        }
    }

    @staticmethod
    def get_pos_weight():
        return torch.tensor([Config.POS_WEIGHT_DICT[Config.DATA_SPLIT_COLUMN][col] for col in Config.LABEL_COLUMNS])

