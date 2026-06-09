import pandas as pd
import numpy as np
import os

from processing_utils import encode_simple_categorical, drop_features_thresh, calculate_fill_rates, drop_baseline, drop_unused_cols, generate_datasets, preprocess_sliding_windows_dataset
from separating_tables_utils import separate_datasets_by_tables

fixed_path = '../donnees_04_26/ALL_DATA_UNION/Fixed/to_train'
sliding_windows_path = '../donnees_04_26/ALL_DATA_UNION/Combined/to_train'
first_symptoms_path = '../donnees_04_26/ALL_DATA_First_Symptoms_UNION/to_train'

csv_files = [f for f in os.listdir(fixed_path) if f.endswith('.csv')]
datasets_sizes = []

for csv_file in csv_files:
    file_path = os.path.join(fixed_path, csv_file)
    
    if os.path.getsize(file_path) == 0:
        print(f"Skipped empty file: {csv_file}")
        continue
    
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Skipped unreadable file: {csv_file}")
        continue
    print(f"Processing {csv_file}")
    separate_datasets_by_tables(df, f'datasets/BEST_MERGE/Fixed/unprocessed/', csv_file)

csv_files = [f for f in os.listdir(sliding_windows_path) if f.endswith('.csv')]
datasets_sizes = []

for csv_file in csv_files:
    file_path = os.path.join(sliding_windows_path, csv_file)
    
    if os.path.getsize(file_path) == 0:
        print(f"Skipped empty file: {csv_file}")
        continue
    
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Skipped unreadable file: {csv_file}")
        continue
    print(f"Processing {csv_file}")
    separate_datasets_by_tables(df, f'datasets/BEST_MERGE/Sliding_windows/unprocessed/', csv_file)

csv_files = [f for f in os.listdir(first_symptoms_path) if f.endswith('.csv')]
datasets_sizes = []

for csv_file in csv_files:
    file_path = os.path.join(first_symptoms_path, csv_file)
    
    if os.path.getsize(file_path) == 0:
        print(f"Skipped empty file: {csv_file}")
        continue
    
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Skipped unreadable file: {csv_file}")
        continue
    print(f"Processing {csv_file}")
    separate_datasets_by_tables(df, f'datasets/BEST_MERGE/First_symptoms/unprocessed/', csv_file)

