import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
log_dir = "tb_logs/MLP_ALSFRS-R_20-11/"
runlog_data = pd.DataFrame({"dataset": [], "trial": [],"mae": [], "rmse": [], "r2": []})
for dataset_name in os.listdir(log_dir):
    dataset_path = os.path.join(log_dir, dataset_name)
    for trial_name in os.listdir(dataset_path):
        trial_path = os.path.join(dataset_path, trial_name)
        for version_name in os.listdir(trial_path):
            version_path = os.path.join(trial_path, version_name)
            try:
                event_acc = EventAccumulator(version_path)
                event_acc.Reload()
                mae = event_acc.Scalars("test_mae")[-1].value
                rmse = event_acc.Scalars("test_rmse")[-1].value
                r2 = event_acc.Scalars("test_r2")[-1].value
                r = {"dataset": [dataset_name], "trial": [trial_name],"mae": [mae], "rmse": [rmse], "r2": [r2]}
                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])
            # Dirty catch of DataLossError
            except Exception:
                print("Event file possibly corrupt: {}".format(version_path))
                traceback.print_exc()

runlog_data.to_csv("stats_entrainement/MLP_ALSFRS-R_20-11/runlog_summary.csv", index=False)