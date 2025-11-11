import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
import optuna
import pandas as pd
import os
import glob
import time

from model import NN
from data import DataModule


datasets_dir = "datasets"
csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))

max_epoch = 200

if torch.backends.mps.is_available():
    accelerator = "mps"
elif torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"

def run_trainings(data_path):
    input_size = len(pd.read_csv(data_path).columns) - 1
    output_size = 1 
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    def objective(trial):
        n_layer = trial.suggest_int('n_layer', 1, 5)
        n_units = trial.suggest_categorical('n_units', [32, 64, 128, 256, 512, 1024])
        decroissant = trial.suggest_categorical('decroissant', [True, False])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        criterion = trial.suggest_categorical('criterion', ['MSE', 'MAE', 'Huber'])
        optimizer = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'Adagrad'])
        activation = trial.suggest_categorical('activation', ['ReLU', 'sigmoid', 'tanh'])


        model = NN(input_size, output_size, n_layer=n_layer, n_units=n_units, learning_rate=learning_rate, decroissant=decroissant, activation=activation, optimizer=optimizer, criterion=criterion)
        dm = DataModule(csv_path=data_path, batch_size=batch_size)

        logger = TensorBoardLogger(f"tb_logs/MLP_ALSFRS-R/{dataset_name}", name=f"trial_{trial.number}")

        trainer = L.Trainer(
            max_epochs=max_epoch,
            accelerator=accelerator,
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
        trainer.test(model, datamodule=dm)
        
        return val_loss
    
    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

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
    L.seed_everything(42, workers=True)

    for csv_file in csv_files:
        print("###########################################")
        print(f"Starting training for dataset: {csv_file}")
        print("###########################################")
        run_trainings(csv_file)