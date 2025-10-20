import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from model import NN
from data import DataModule
from config import INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DATA_PATH, DEVICE

if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="my_model")

    model = NN(INPUT_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE)
    dm = DataModule(csv_path=DATA_PATH, batch_size=BATCH_SIZE)
    trainer = L.Trainer(max_epochs=EPOCHS, logger=logger, accelerator=DEVICE.type)

    logger.experiment.add_text("model/hyperparameters",
                               f"batch_size: {BATCH_SIZE}, learning_rate: {LEARNING_RATE}, epochs: {EPOCHS}")
    logger.experiment.add_text("model/architecture", str(model))
    logger.experiment.add_scalar("model/parameters", sum(p.numel() for p in model.parameters()))

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)