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
# - Dossier où sauvegarder les poids
def run_trainings(data_path, log_dir="MLP_regression/tb_logs/", weights_dir="MLP_regression/weights/", study_name="mlp_regression", trials=100, trial_epoch=30, max_epoch=300):
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

        # Initialisation du modèle avec les paramètres choisis (voir model.py
        model = MLP_regressor(input_size, output_dim=1, n_layer=n_layer, n_units=n_units, learning_rate=learning_rate, decroissant=decroissant, activation='ReLU', optimizer='Adam', criterion='Huber')

        # Dataset et dataloader (voir data.py)
        dm = DataModule(csv_path=data_path, batch_size=batch_size)

        # Prunning des trials qui convergent trop lentement par rapport aux autres
        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss"
        )

        # Fonction qui execute la boucle d'entrainement
        trainer = L.Trainer(
            max_epochs=trial_epoch, # Nombre d'epoch maximum
            accelerator=accelerator, # GPU si possible
            callbacks=[EarlyStopping(monitor='val_loss', patience=5), # Early stopping 
                       pruning_callback], # Prunning
            enable_checkpointing=False,
            logger=False # Pas de log pour les trials, déjà dans optuna.db
        )

        # Entrainement du trial et validation
        trainer.fit(model, dm)
        val_result = trainer.validate(model, datamodule=dm) 
        val_loss = val_result[0]['val_loss']
        
        return val_loss # Retourne la perte du dataset de validation
    
    start_time = time.time() # Chronomètre

    # Optimisation
    study = optuna.create_study(
        storage=f"sqlite:///optuna.db", # Stockage de l'étude dans une base de données pour visualisation
        study_name=f"{study_name}_{dataset_name}", # Nom de l'étude
        load_if_exists=True, # Si l'étude crash elle peut reprendre là où elle s'était arrêtée
        direction='minimize') # On cherche à minimiser la sortie de objective donc la perte de la validation
    study.optimize(objective, n_trials=trials) # trials

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

    # Entrainement du modèle avec les meilleurs paramètres sur le nombre d'epoch maximum
    best_params = trial.params
    best_model = MLP_regressor(input_size, 
                               output_dim=1, 
                               n_layer=best_params['n_layer'], 
                               n_units=best_params['n_units'], 
                               learning_rate=best_params['learning_rate'], 
                               decroissant=best_params['decroissant'], 
                               activation='ReLU', 
                               optimizer='Adam', 
                               criterion='Huber')
    
    dm = DataModule(csv_path=data_path, batch_size=best_params['batch_size'])

    trainer = L.Trainer(
        max_epochs=max_epoch, # Nombre d'epoch maximum
        accelerator=accelerator, # GPU si possible
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)], # Early stopping 
        logger=TensorBoardLogger(f"{log_dir}{dataset_name}", name=f"best_trial"), # Log
        enable_checkpointing=False
    )

    trainer.fit(best_model, dm) # Entrainement
    trainer.test(best_model, datamodule=dm) # Test

    # Sauvegarder les poids
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{dataset_name}.pt")
    torch.save(best_model.state_dict(), weights_path)
    print(f"Poids sauvegardés → {weights_path}")

if __name__ == '__main__':
    csv_combined = glob.glob(os.path.join("datasets/ALSFRS_R_COMBINED", "*.csv"))
    csv_fixed = glob.glob(os.path.join("datasets/ALSFRS_R_FIXED", "*.csv"))
    csv_first_symptoms = glob.glob(os.path.join("datasets/ALSFRS_R_F_S_FIXED", "*.csv"))

    trials = 100
    trial_epoch = 200
    max_epoch = 300

    # for csv_file in csv_combined:
    #     print(f"Training on dataset: {csv_file}")

    #     L.seed_everything(42)

    #     run_trainings(csv_file, 
    #                   log_dir="MLP_regression/tb_logs/ALSFRS_R_COMBINED/",
    #                   weights_dir="MLP_regression/weights/ALSFRS_R_COMBINED/",
    #                   trials=trials,
    #                   trial_epoch=trial_epoch,
    #                   max_epoch=max_epoch)

    # for csv_file in csv_fixed:
    #     print(f"Training on dataset: {csv_file}")

    L.seed_everything(42)

    run_trainings("datasets/ALSFRS_R_F_S_FIXED/ALSFRS_R_F_S_FIXED_6_9M.csv", 
                    log_dir="MLP_regression/tb_logs/ALSFRS_R_FIXED/Test",
                    weights_dir="MLP_regression/weights/ALSFRS_R_FIXED/Test",
                    study_name="mlp_regression_fixed_test",
                    trials=trials,
                    trial_epoch=trial_epoch,
                    max_epoch=max_epoch)

    # for csv_file in csv_first_symptoms:
    #     print(f"Training on dataset: {csv_file}")

    #     L.seed_everything(42)

    #     run_trainings(csv_file, 
    #                   log_dir="MLP_regression/tb_logs/ALSFRS_R_F_S_FIXED/",
    #                   weights_dir="MLP_regression/weights/ALSFRS_R_F_S_FIXED/",
    #                   trials=trials,
    #                   trial_epoch=trial_epoch,
    #                   max_epoch=max_epoch)