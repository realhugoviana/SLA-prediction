import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf  # TensorBoard tensor protobufs utilisent TF

def get_hyperparameters_from_tensor(trial_path):
    ea = EventAccumulator(trial_path)
    ea.Reload()

    tags = ea.Tags()
    if "tensors" not in tags or 'hyperparameters/text_summary' not in tags['tensors']:
        raise ValueError("No 'hyperparameters/text_summary' tensor found in this trial.")

    # Récupère le tensor
    tensor_events = ea.Tensors('hyperparameters/text_summary')
    if not tensor_events:
        raise ValueError("No tensor events found for 'hyperparameters/text_summary'.")

    # Normalement, le premier tensor contient le texte
    tensor_proto = tensor_events[0].tensor_proto
    text_bytes = tensor_proto.string_val[0]  # bytes
    text = text_bytes.decode('utf-8')

    # Parse "key: value, key: value, ..."
    hp_dict = {}
    for item in text.split(","):
        key, value = item.split(":")
        hp_dict[key.strip()] = value.strip()

    return hp_dict


# ------- Exemple -------
log_dir = "tb_logs/MLP_ALSFRS-R_RANDOM/"
all_data = []
for subdir in os.listdir(log_dir):
    subdir_path = os.path.join(log_dir, subdir)
    for dataset in os.listdir(subdir_path):
        dataset_path = os.path.join(subdir_path, dataset)
        for trial_name in os.listdir(dataset_path):
            trial_path = os.path.join(dataset_path, trial_name)
            for version_name in os.listdir(trial_path):
                version_path = os.path.join(trial_path, version_name)
                hyperparams = get_hyperparameters_from_tensor(version_path)

                hp_dict = {
                    "dataset": dataset,
                    "trial": trial_name,
                }
                for k, v in hyperparams.items():
                    hp_dict[k] = v

                all_data.append(hp_dict)

df_hyperparams = pd.DataFrame(all_data)
df_hyperparams.to_csv("stats_entrainement/batch_size.csv", index=False)
                        

# trial_path = "tb_logs/MLP_ALSFRS-R_RANDOM/T2/MLP_alsfrs-r_T1_T2/trial_0/version_0"
# hyperparams = get_hyperparameters_from_tensor(trial_path)

# print("\nHyperparameters for this trial:\n")
# for k, v in hyperparams.items():
#     print(f"{k} -> {v}")