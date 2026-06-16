import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def log_to_csv(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    runlog_data = pd.DataFrame({"dataset": [], 
                                "fold": [],
                                "mae": [], 
                                "rmse": [], 
                                "r2": []})
    for dataset_name in os.listdir(log_dir):
        dataset_path = os.path.join(log_dir, dataset_name)
        for fold_index in os.listdir(dataset_path):
            fold_path = os.path.join(dataset_path, fold_index)
            for version_name in os.listdir(fold_path):
                version_path = os.path.join(fold_path, version_name)
                try:
                    event_acc = EventAccumulator(version_path)
                    event_acc.Reload()
                    
                    tags = event_acc.Tags()["scalars"]
                    if "test_mae" not in tags:
                        continue

                    mae = event_acc.Scalars("test_mae")[-1].value
                    rmse = event_acc.Scalars("test_rmse")[-1].value
                    r2 = event_acc.Scalars("test_r2")[-1].value

                    r = {"dataset": [dataset_name], 
                        "fold": [fold_index],
                        "mae": [mae], 
                        "rmse": [rmse], 
                        "r2": [r2]}
                    r = pd.DataFrame(r)
                    runlog_data = pd.concat([runlog_data, r])
                # Dirty catch of DataLossError
                except Exception:
                    print("Event file possibly corrupt: {}".format(version_path))
                    traceback.print_exc()



    runlog_data.to_csv(f'{output_dir}/runlog_summary.csv', index=False)

    runlog_data = pd.read_csv(f'{output_dir}/runlog_summary.csv')

    # Fonction pour calculer l'intervalle de confiance 95%
    def calculate_ci(series):
        mean = series.mean()
        std = series.std()
        n = len(series)
        # Utilisation du z-score critique (1.96) pour un niveau de confiance de 95%, 
        # adapté à une grande taille d'échantillon (N).
        margin_of_error = 1.96 * std / (n**0.5)
        return mean - margin_of_error, mean + margin_of_error

    # Calcul des statistiques pour chaque dataset
    summary_stats = runlog_data.groupby('dataset').agg(
        mae_mean=('mae', 'mean'),
        mae_IC_95_low=('mae', lambda x: calculate_ci(x)[0]), # CI bas
        mae_IC_95_high=('mae', lambda x: calculate_ci(x)[1]),# CI haut
        mae_std=('mae', 'std'),

        rmse_mean=('rmse', 'mean'),
        rmse_IC_95_low=('rmse', lambda x: calculate_ci(x)[0]), 
        rmse_IC_95_high=('rmse', lambda x: calculate_ci(x)[1]),
        rmse_std=('rmse', 'std'),

        r2_mean=('r2', 'mean'),
        r2_IC_95_low=('r2', lambda x: calculate_ci(x)[0]),
        r2_IC_95_high=('r2', lambda x: calculate_ci(x)[1]),
        r2_std=('r2', 'std')
    )

    # Sauvegarde des statistiques agrégées
    summary_stats.to_csv(f'{output_dir}/statistical_summary_by_dataset.csv') 

log_dir = "MLP_regression/tb_logs/mlp_sythetic/"
output_dir = "MLP_regression/stats_entrainement/mlp_sythetic/"

log_to_csv(log_dir, output_dir)