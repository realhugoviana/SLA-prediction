import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
import optuna

from model import NN
from data import DataModule
from config import INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, DATA_PATH, DEVICE

def objective(trial):
    n_layer = trial.suggest_int('n_layer', 1, 5)
    n_units = trial.suggest_categorical('n_units', [16, 32, 64, 128, 256])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])


    model = NN(INPUT_SIZE, OUTPUT_SIZE, n_layer=n_layer, n_units=n_units, learning_rate=learning_rate)
    dm = DataModule(csv_path=DATA_PATH, batch_size=batch_size)

    logger = TensorBoardLogger("tb_logs", name=f"trial_{trial.number}")

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE.type,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
        logger=logger,
        enable_checkpointing=False
    )

    logger.experiment.add_text("hyperparameters", 
        f"batch_size: {batch_size}, learning_rate: {learning_rate}, n_layer: {n_layer}, n_units: {n_units}")
    logger.experiment.add_text("architecture", str(model))
    logger.experiment.add_scalar("parameters", sum(p.numel() for p in model.parameters()))

    trainer.fit(model, dm)
    val_result = trainer.validate(model, datamodule=dm)
    val_loss = val_result[0]['val_loss']
    
    return val_loss

if __name__ == '__main__':
    L.seed_everything(42, workers=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# if __name__ == '__main__':
#     logger = TensorBoardLogger("tb_logs", name="my_model")

#     model = NN(INPUT_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE)
#     dm = DataModule(csv_path=DATA_PATH, batch_size=BATCH_SIZE)
#     trainer = L.Trainer(max_epochs=EPOCHS, logger=logger, accelerator=DEVICE.type)

#     logger.experiment.add_text("hyperparameters", f"batch_size: {BATCH_SIZE}, learning_rate: {LEARNING_RATE}, epochs: {EPOCHS}")
#     logger.experiment.add_text("architecture", str(model))
#     logger.experiment.add_scalar("parameters", sum(p.numel() for p in model.parameters()))

#     trainer.fit(model, dm)
#     trainer.test(model, datamodule=dm)
#     trainer.validate(model, datamodule=dm)