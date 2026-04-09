
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

log_dir = "MLP_regression/tb_logs/MLP_ALSFRS-R_COMBINED/"
runlog_data = pd.DataFrame({"dataset": [], 
                            "trial": [], 
                            "n_layer": [], 
                            "n_units": [], 
                            "learning_rate": [],
                            "decroissant": [],
                            "batch_size": [],
                            "activation": [],
                            "optimizer": [],
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

                reader = SummaryReader(version_path)
                hp = reader.hparams
                hp_dict = dict(zip(hp["tag"], hp["value"]))

                mae = event_acc.Scalars("test_mae")[-1].value
                rmse = event_acc.Scalars("test_rmse")[-1].value
                r2 = event_acc.Scalars("test_r2")[-1].value

                batch_size = get_batch_size(version_path)["batch_size"]

                n_layer = hp_dict["n_layer"]
                n_units = hp_dict["n_units"]
                learning_rate = hp_dict["learning_rate"]
                decroissant = hp_dict["decroissant"]
                activation = hp_dict["activation"]
                optimizer = hp_dict["optimizer"]

                r = {"dataset": [dataset_name], 
                     "trial": [trial_name],
                     "n_layer": [n_layer], 
                     "n_units": [n_units], 
                     "learning_rate": [learning_rate],
                     "decroissant": [decroissant],
                     "batch_size": [batch_size],
                     "activation": [activation],
                     "optimizer": [optimizer],
                     "mae": [mae], 
                     "rmse": [rmse], 
                     "r2": [r2]}
                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])
            # Dirty catch of DataLossError
            except Exception:
                print("Event file possibly corrupt: {}".format(version_path))
                traceback.print_exc()

runlog_data["dataset"] = runlog_data["dataset"].str.replace("MLP_alsfrs-r_", "")
runlog_data.to_csv("MLP_regression/stats_entrainement/MLP_ALSFRS-R_COMBINED_15_03/runlog_summary.csv", index=False)