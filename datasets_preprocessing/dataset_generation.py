import pandas as pd
import numpy as np
import os

from processing_utils import calculate_fill_rates, generate_datasets, preprocess_sliding_windows_dataset


# Preprocesses combined datasets from all csv files in a folder and saves them to csv files
# Input: String path to the folder containing the csv files, String prefix for the generated csv files
# Output: Pandas Dataframe
def preprocess_sliding_windows_datasets_from_folder(folder_path, file_prefix, drop_feat=True, thresh=80, baseline=False):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    datasets_sizes = pd.DataFrame({"dataset": [], 
                    "size": []})
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        if os.path.getsize(file_path) == 0:
            print(f"Skipped empty file: {csv_file}")
            continue
        
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Skipped unreadable file: {csv_file}")
            continue
        
        calculate_fill_rates(df).to_csv(f'{file_prefix}fill_rate_{csv_file}', index=False)
        
        output_file_name = f'{file_prefix}{csv_file}'
        print(f"Processing {csv_file}")
        df = preprocess_sliding_windows_dataset(df, output_file_name, drop_feat=drop_feat, thresh=thresh, baseline=baseline)

        datasets_sizes = pd.concat([datasets_sizes, pd.DataFrame({"dataset": [output_file_name], 
                                                                  "size": [len(df)],
                                                                  "nb_cols": [len(df.columns)],
                                                                  "nb_cols_non_als": [len(df.columns) - sum(
                                                                      1 for c in df.columns
                                                                      if c.startswith('ALS') or c == 'Target'
                                                                  )]
                                                                  })])
    
    return datasets_sizes

# Generates datasets from all csv files in a folder and saves them to csv files, also outputs a dataframe of the size of the generated datasets
# Input: String path to the folder containing the csv files, String prefix for the generated csv files
# Output: Pandas Dataframe
def generate_datasets_from_folder(folder_path, file_prefix, drop_feat=True, thresh=80, baseline=False):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    datasets_sizes = []
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        if os.path.getsize(file_path) == 0:
            print(f"Skipped empty file: {csv_file}")
            continue
        
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Skipped unreadable file: {csv_file}")
            continue
        print(f"Processing {csv_file}")
        datasets_size = generate_datasets(df, file_prefix, drop_feat=drop_feat, thresh=thresh, baseline=baseline)
        datasets_sizes.append(datasets_size)
    
    return pd.concat(datasets_sizes, ignore_index=True)

fixed_path = 'datasets/BEST_MERGE/Fixed/unprocessed/'
sliding_windows_path = 'datasets/BEST_MERGE/Sliding_windows/unprocessed/'
first_symptoms_path = 'datasets/BEST_MERGE/First_symptoms/unprocessed/'


for table_path in os.scandir(fixed_path):
    table = os.path.basename(table_path)

    if table == ".DS_Store":
        continue
    
    os.makedirs(os.path.dirname(f'datasets/BEST_MERGE/Fixed/to_train/{table}/'), exist_ok=True)
    if table == "VIT":
        datasets_size_Fixed = generate_datasets_from_folder(table_path, f'datasets/BEST_MERGE/Fixed/to_train/{table}/', drop_feat=True, thresh=70, baseline=False)
    else:
        datasets_size_Fixed = generate_datasets_from_folder(table_path, f'datasets/BEST_MERGE/Fixed/to_train/{table}/', drop_feat=True, thresh=80, baseline=False)
    datasets_size_Fixed.to_csv(f'datasets/BEST_MERGE_{table}_Fixed_datasets_size.csv', index=False)

for table_path in os.scandir(sliding_windows_path):
    table = os.path.basename(table_path)

    if table == ".DS_Store":
        continue

    os.makedirs(os.path.dirname(f'datasets/BEST_MERGE/Sliding_windows/to_train/{table}/'), exist_ok=True)
    if table == "VIT":
        datasets_size_Sliding_Windows = preprocess_sliding_windows_datasets_from_folder(table_path, f'datasets/BEST_MERGE/Sliding_windows/to_train/{table}/', drop_feat=True, thresh=70, baseline=False)
    else:
        datasets_size_Sliding_Windows = preprocess_sliding_windows_datasets_from_folder(table_path, f'datasets/BEST_MERGE/Sliding_windows/to_train/{table}/', drop_feat=True, thresh=80, baseline=False)
    datasets_size_Sliding_Windows.to_csv(f'datasets/BEST_MERGE_{table}_Sliding_windows_datasets_size.csv', index=False)

for table_path in os.scandir(first_symptoms_path):
    table = os.path.basename(table_path)

    if table == ".DS_Store":
        continue

    os.makedirs(os.path.dirname(f'datasets/BEST_MERGE/First_symptoms/to_train/{table}/'), exist_ok=True)
    if table == "VIT":
        datasets_size_F_S_Fixed = generate_datasets_from_folder(table_path, f'datasets/BEST_MERGE/First_symptoms/to_train/{table}/', drop_feat=True, thresh=70, baseline=False)
    else:
        datasets_size_F_S_Fixed = generate_datasets_from_folder(table_path, f'datasets/BEST_MERGE/First_symptoms/to_train/{table}/', drop_feat=True, thresh=80, baseline=False)
    datasets_size_F_S_Fixed.to_csv(f'datasets/BEST_MERGE_{table}_First_symptoms_datasets_size.csv', index=False)
