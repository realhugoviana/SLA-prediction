import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def log_to_csv(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    runlog_data = pd.DataFrame({"dataset": [], 
                                "fold": []})
    
    for i in range(1, 13):
        runlog_data[f"mae_{i}"] = []
        runlog_data[f"rmse_{i}"] = []
        runlog_data[f"r2_{i}"] = []

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
                    if "test_mae_1" not in tags:
                        continue

                    r = {"dataset": [dataset_name], 
                        "fold": [fold_index]}

                    for i in range(1, 14):
                        r[f"mae_{i}"] = [event_acc.Scalars(f"test_mae_{i}")[-1].value]
                        r[f"rmse_{i}"] = [event_acc.Scalars(f"test_rmse_{i}")[-1].value]
                        r[f"r2_{i}"] = [event_acc.Scalars(f"test_r2_{i}")[-1].value]

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

    summary_stats = pd.DataFrame()
    for i in range(1, 13):
        # Calcul des statistiques pour chaque dataset
        summary_stats[f'mae_mean_{i}'] = runlog_data.groupby('dataset').agg(mae_mean=(f'mae_{i}', 'mean'))
        summary_stats[f'mae_IC_95_low_{i}'] = runlog_data.groupby('dataset').agg(mae_IC_95_low=(f'mae_{i}', lambda x: calculate_ci(x)[0]))
        summary_stats[f'mae_IC_95_high_{i}'] = runlog_data.groupby('dataset').agg(mae_IC_95_high=(f'mae_{i}', lambda x: calculate_ci(x)[1]))
        summary_stats[f'mae_std_{i}'] = runlog_data.groupby('dataset').agg(mae_std=(f'mae_{i}', 'std'))

        summary_stats[f'rmse_mean_{i}'] = runlog_data.groupby('dataset').agg(rmse_mean=(f'rmse_{i}', 'mean'))
        summary_stats[f'rmse_IC_95_low_{i}'] = runlog_data.groupby('dataset').agg(rmse_IC_95_low=(f'rmse_{i}', lambda x: calculate_ci(x)[0]))
        summary_stats[f'rmse_IC_95_high_{i}'] = runlog_data.groupby('dataset').agg(rmse_IC_95_high=(f'rmse_{i}', lambda x: calculate_ci(x)[1]))
        summary_stats[f'rmse_std_{i}'] = runlog_data.groupby('dataset').agg(rmse_std=(f'rmse_{i}', 'std'))
        
        summary_stats[f'r2_mean_{i}'] = runlog_data.groupby('dataset').agg(r2_mean=(f'r2_{i}', 'mean'))
        summary_stats[f'r2_IC_95_low_{i}'] = runlog_data.groupby('dataset').agg(r2_IC_95_low=(f'r2_{i}', lambda x: calculate_ci(x)[0]))
        summary_stats[f'r2_IC_95_high_{i}'] = runlog_data.groupby('dataset').agg(r2_IC_95_high=(f'r2_{i}', lambda x: calculate_ci(x)[1]))
        summary_stats[f'r2_std_{i}'] = runlog_data.groupby('dataset').agg(r2_std=(f'r2_{i}', 'std'))
    # Sauvegarde des statistiques agrégées
    summary_stats.to_csv(f'{output_dir}/statistical_summary_by_dataset.csv') 

log_dir = "RNN/tb_logs/gru_synthetic_noisy_sigmoid/"
output_dir = "RNN/stats_entrainement/gru_synthetic_noisy_sigmoid/"

log_to_csv(log_dir, output_dir)

log_dir = "RNN/tb_logs/rnn_synthetic_noisy_sigmoid/"
output_dir = "RNN/stats_entrainement/rnn_synthetic_noisy_sigmoid/"

log_to_csv(log_dir, output_dir)

log_dir = "RNN/tb_logs/lstm_synthetic_noisy_sigmoid/"
output_dir = "RNN/stats_entrainement/lstm_synthetic_noisy_sigmoid/"

log_to_csv(log_dir, output_dir)