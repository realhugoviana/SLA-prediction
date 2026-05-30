import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tbparse import SummaryReader

def get_batch_size(trial_path):
    ea = EventAccumulator(trial_path)
    ea.Reload()

    tags = ea.Tags()
    if "tensors" not in tags or 'hyperparameters/text_summary' not in tags['tensors']:
        raise ValueError("No 'hyperparameters/text_summary' tensor found in this trial.")

    tensor_events = ea.Tensors('hyperparameters/text_summary')
    if not tensor_events:
        raise ValueError("No tensor events found for 'hyperparameters/text_summary'.")

    tensor_proto = tensor_events[0].tensor_proto
    text_bytes = tensor_proto.string_val[0]
    text = text_bytes.decode('utf-8')

    hp_dict = {}
    for item in text.split(","):
        key, value = item.split(":")
        if key.strip() == "batch_size":
            hp_dict[key.strip()] = value.strip()

    return hp_dict

def log_to_csv(log_dir, output_dir):
    runlog_data = pd.DataFrame({"dataset": [], 
                                "trial": [],
                                "mae": [], 
                                "rmse": [], 
                                "r2": []})
    for dataset_name in os.listdir(log_dir):
        dataset_path = os.path.join(log_dir, dataset_name)
        for trial_name in os.listdir(dataset_path):
            trial_path = os.path.join(dataset_path, trial_name)
            for version_name in os.listdir(trial_path):
                version_path = os.path.join(trial_path, version_name)
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
                        "trial": [trial_name],
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
        mae_min=('mae', 'min'),
        mae_max=('mae', 'max'),
        mae_ci_low=('mae', lambda x: calculate_ci(x)[0]), # CI bas
        mae_ci_high=('mae', lambda x: calculate_ci(x)[1]),# CI haut

        rmse_mean=('rmse', 'mean'),
        rmse_min=('rmse', 'min'),
        rmse_max=('rmse', 'max'),
        rmse_ci_low=('rmse', lambda x: calculate_ci(x)[0]), 
        rmse_ci_high=('rmse', lambda x: calculate_ci(x)[1]),

        r2_mean=('r2', 'mean'),
        r2_min=('r2', 'min'),
        r2_max=('r2', 'max'),
        r2_ci_low=('r2', lambda x: calculate_ci(x)[0]),
        r2_ci_high=('r2', lambda x: calculate_ci(x)[1])
    )

    # Sauvegarde des statistiques agrégées
    summary_stats.to_csv(f'{output_dir}/statistical_summary_by_dataset.csv') 


log_dir = "MLP_regression/tb_logs/ALSFRS_R_COMBINED_27_05/"
output_dir = "MLP_regression/stats_entrainement/ALSFRS_R_COMBINED_27_05/"

log_to_csv(log_dir, output_dir)

log_dir = "MLP_regression/tb_logs/ALSFRS_R_FIXED_27_05/"
output_dir = "MLP_regression/stats_entrainement/ALSFRS_R_FIXED_27_05/"

log_to_csv(log_dir, output_dir)