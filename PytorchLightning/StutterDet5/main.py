from pprint import pprint
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config import Config
from StutterDetModel.PytorchLightning.StutterDet5.dataset import StutterDataModule
from model import Wav2Vec2Model

def main():
    pl.seed_everything(100)
    torch.set_float32_matmul_precision("high")

    datamodule = StutterDataModule()
    model = Wav2Vec2Model()
    
    # logger = TensorBoardLogger(save_dir=Config.LOG_DIR)
    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     dirpath=Config.SAVED_MODEL_DIR,
    #     filename='w2v2-{epoch:02d}-{val_loss:.3f}-'+str(timestamp),
    #     mode='min',
    # )
    # earlystopping_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=3,
    #     verbose=True,
    #     mode="min"
    # )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        max_epochs=Config.N_EPOCHS,
        # logger=logger,
        # callbacks=[
        #     checkpoint_callback,
        #     earlystopping_callback
        # ],
        overfit_batches=10,
    )

    trainer.fit(
        model=model, 
        datamodule=datamodule
    )

    # trainer.validate(
    #     model=model,
    #     datamodule=sep28k_datamodule
    # )

    # trainer.test(
    #     model=model,
    #     dataloaders=sep28k_datamodule
    # )

if __name__ == "__main__":

    print("Parameters being used are: ")
    pprint(Config.__dict__)
    print("-"*80)

    main()
    
    print("End")