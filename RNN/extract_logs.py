import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def log_to_csv(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    runlog_data = pd.DataFrame({"fold": []})
    
    for i in range(1, 14):
        runlog_data[f"mae_{i}"] = []
        runlog_data[f"rmse_{i}"] = []
        runlog_data[f"r2_{i}"] = []

    for fold_index in os.listdir(log_dir):
        fold_path = os.path.join(log_dir, fold_index)
        for version_name in os.listdir(fold_path):
            version_path = os.path.join(fold_path, version_name)
            try:
                event_acc = EventAccumulator(version_path)
                event_acc.Reload()
                
                tags = event_acc.Tags()["scalars"]
                if "test_mae_1" not in tags:
                    continue

                r = {"fold": [fold_index]}

                for i in range(1, 14):
                    r[f"mae_{i}"] = [event_acc.Scalars(f"test_mae_{i}")[-1].value]
                    r[f"rmse_{i}"] = [event_acc.Scalars(f"test_rmse_{i}")[-1].value]
                    r[f"r2_{i}"] = [event_acc.Scalars(f"test_r2_{i}")[-1].value]

                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])
                print(r)
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

    summary_stats = pd.DataFrame({"month": ["3", "6", "9", "12"]})


    # Calcul des statistiques pour chaque dataset
    # summary_stats[f'{i}'] = runlog_data.agg(mae_mean=(f'mae_{i}', 'mean'))
    summary_stats['mae_mean'] = [runlog_data[f"mae_3"].mean(), 
                                 runlog_data[f"mae_6"].mean(),
                                 runlog_data[f"mae_9"].mean(),
                                 runlog_data[f"mae_12"].mean()]
    
    summary_stats['mae_std'] = [runlog_data[f"mae_3"].std(), 
                                runlog_data[f"mae_6"].std(),
                                runlog_data[f"mae_9"].std(),
                                runlog_data[f"mae_12"].std()]
    

    summary_stats['rmse_mean'] = [runlog_data[f"rmse_3"].mean(), 
                                runlog_data[f"rmse_6"].mean(),
                                runlog_data[f"rmse_9"].mean(),
                                runlog_data[f"rmse_12"].mean()]
    
    summary_stats['rmse_std'] = [runlog_data[f"rmse_3"].std(), 
                                runlog_data[f"rmse_6"].std(),
                                runlog_data[f"rmse_9"].std(),
                                runlog_data[f"rmse_12"].std()]


    summary_stats['r2_mean'] = [runlog_data[f"r2_3"].mean(), 
                                runlog_data[f"r2_6"].mean(),
                                runlog_data[f"r2_9"].mean(),
                                runlog_data[f"r2_12"].mean()]
        
    summary_stats['r2_std'] = [runlog_data[f"r2_3"].std(), 
                                runlog_data[f"r2_6"].std(),
                                runlog_data[f"r2_9"].std(),
                                runlog_data[f"r2_12"].std()]
    #Sauvegarde des statistiques agrégées
    summary_stats.to_csv(f'{output_dir}/statistical_summary_by_dataset.csv') 

log_dir = "RNN/tb_logs/lstm_interpolation_multitask/sliding_windows/"
output_dir = "RNN/stats_entrainement/lstm_interpolation_multitask/"

log_to_csv(log_dir, output_dir)