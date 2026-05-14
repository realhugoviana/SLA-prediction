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

# Permets d'utiliser le GPU
if torch.backends.mps.is_available(): # MPS pour mac
    accelerator = "mps"
elif torch.cuda.is_available(): # Cuda sinon
    accelerator = "gpu"
else:
    accelerator = "cpu" # CPU si Cuda pas supporté

# Fonction d'optimisation des paramètres et d'entrainement
# Entrée : 
# - Chemin d'accès au csv
# - Dossier où mettre les logs
def run_trainings(data_path, log_dir="MLP_regression/tb_logs/"):
    input_size = len(pd.read_csv(data_path).columns) - 1 # Nombre de colonnes - target
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    # Fonction d'optimisation Optuna
    def objective(trial):

        # Paramètres optimisés
        n_layer = trial.suggest_int('n_layer', 1, 3)
        n_units = trial.suggest_categorical('n_units', [32, 64, 128, 256, 512])
        decroissant = trial.suggest_categorical('decroissant', [True, False])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        criterion = trial.suggest_categorical('criterion', ['MSE', 'MAE', 'Huber'])
        optimizer = trial.suggest_categorical('optimizer', ['Adam'])
        activation = trial.suggest_categorical('activation', ['ReLU', 'tanh', 'sigmoid'])

        # Initialisation du modèle avec les paramètres choisis (voir model.py)
        model = MLP_regressor(input_size, output_dim=1, n_layer=n_layer, n_units=n_units, learning_rate=learning_rate, decroissant=decroissant, activation=activation, optimizer=optimizer, criterion=criterion)

        # Dataset et dataloader (voir data.py)
        dm = DataModule(csv_path=data_path, batch_size=batch_size)

        # Log de chaque trial
        logger = TensorBoardLogger(f"{log_dir}{dataset_name}", name=f"trial_{trial.number}")

        # Prunning des trials qui convergent trop lentement par rapport aux autres
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss"
        )

        # Fonction qui execute la boucle d'entrainement
        trainer = L.Trainer(
            max_epochs=max_epoch, # Nombre d'epoch maximum
            accelerator=accelerator, # GPU si possible
            callbacks=[EarlyStopping(monitor='val_loss', patience=5), # Early stopping 
                       pruning_callback], # Prunning
            logger=logger, # Log
            enable_checkpointing=False # Ne stocke pas les poids
        )

        # Log les paramètres
        logger.experiment.add_text("hyperparameters", 
            f"batch_size: {batch_size}, learning_rate: {learning_rate}, n_layer: {n_layer}, n_units: {n_units}")
        logger.experiment.add_text("architecture", str(model))
        logger.experiment.add_scalar("parameters", sum(p.numel() for p in model.parameters()))

        # Entrainement du trial, validation et test
        trainer.fit(model, dm)
        val_result = trainer.validate(model, datamodule=dm) 
        val_loss = val_result[0]['val_loss']
        trainer.test(model, datamodule=dm)
        
        return val_loss # Retourne la perte du dataset de validation
    
    start_time = time.time() # Chronomètre

    # Optimisation
    study = optuna.create_study(direction='minimize') # On cherche à minimiser la sortie de objective donc la perte de la validation
    study.optimize(objective, n_trials=100) # 100 trials

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds for dataset {dataset_name}.")

    # Stock les meilleurs paramètres
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    csv_combined = glob.glob(os.path.join("datasets/ALSFRS_R_COMBINED", "*.csv"))
    csv_fixed = glob.glob(os.path.join("datasets/ALSFRS_R_FIXED", "*.csv"))
    csv_first_symptoms = glob.glob(os.path.join("datasets/ALSFRS_R_F_S_FIXED", "*.csv"))

    max_epoch = 200

    for csv_file in csv_combined:
        print(f"Training on dataset: {csv_file}")

        L.seed_everything(42)

        run_trainings(csv_file, log_dir="MLP_regression/tb_logs/ALSFRS_R_COMBINED/")

    for csv_file in csv_fixed:
        print(f"Training on dataset: {csv_file}")

        L.seed_everything(42)

        run_trainings(csv_file, log_dir="MLP_regression/tb_logs/ALSFRS_R_FIXED/")

    for csv_file in csv_first_symptoms:
        print(f"Training on dataset: {csv_file}")

        L.seed_everything(42)

        run_trainings(csv_file, log_dir="MLP_regression/tb_logs/ALSFRS_R_F_S_FIXED/")