import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
import optuna
import pandas as pd
import os
import glob
import time

from model import MLP_regressor
from data import DataModule


if torch.backends.mps.is_available():
    accelerator = "mps"
elif torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"

def run_trainings(data_path):
    input_size = len(pd.read_csv(data_path).columns) - 1
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    def objective(trial):
        n_layer = trial.suggest_int('n_layer', 1, 3)
        n_units = trial.suggest_categorical('n_units', [32, 64, 128, 256, 512])
        decroissant = trial.suggest_categorical('decroissant', [True, False])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        optimizer = trial.suggest_categorical('optimizer', ['Adam'])
        activation = trial.suggest_categorical('activation', ['ReLU'])


        model = MLP_regressor(input_size, n_layer=n_layer, n_units=n_units, learning_rate=learning_rate, decroissant=decroissant, activation=activation, optimizer=optimizer)
        dm = DataModule(csv_path=data_path, batch_size=batch_size)

        logger = TensorBoardLogger(f"MLP_regressor/tb_logs/MLP_ALSFRS-R/{dataset_name}", name=f"trial_{trial.number}")

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss"
        )

        trainer = L.Trainer(
            max_epochs=max_epoch,
            accelerator=accelerator,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5), pruning_callback],
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
        trainer.test(model, datamodule=dm)
        
        return val_loss
    
    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds for dataset {dataset_name}.")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':

    datasets_dir = "../datasets/to_train"
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))

    max_epoch = 200

    for csv_file in csv_files:
        print(f"Training on dataset: {csv_file}")

        L.seed_everything(42)

        run_trainings(csv_file)