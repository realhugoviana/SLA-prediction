import pandas as pd
import numpy as np
import os

from best_performing_merge_processing import encode_simple_categorical, drop_features_thresh, drop_features_by_features, r2_eta2, calculate_fill_rates

# Drops questions 10 and ALSFRS scores columns (not ALSFRS-R score)
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def drop_alsfrs_cols(df):
    cols_to_drop = df.columns[
        df.columns.str.contains(
            r'ALS_ALSFRS_Total|ALS_Q10|Test_Unit'
        )
    ]

    df = df.drop(columns=cols_to_drop)
    df = df.drop("subject_id", axis=1, errors='ignore')

    return df

# Drops features with more than a certain percentage of missing values and keeps only features with the highest r2/eta2 with the target variable
# Input: Pandas Dataframe, Threshold percentage of missing values
# Output: Pandas Dataframe
def drop_features(df, thresh):
    # cols_r2_eta2 = r2_eta2(drop_features_thresh(df, df, thresh=thresh))["col"]

    # for col in cols_r2_eta2:
    #     print(f"Colonne {col} : r2/eta2 = {cols_r2_eta2[col].iloc[0]:.4f}%")

    # df_dropped = drop_features_by_features(df, cols_r2_eta2.index[:100])
    df_dropped = drop_features_thresh(df, df, thresh=thresh)

    # for col in df_dropped.columns:
        # if not col.startswith('ALS_') and col != 'Target':
            # print(f"Colonne {col} conservée (r2/eta2 = {cols_r2_eta2[col].iloc[0]:.4f}%)")
            # print(f"Colonne {col} conservée")
    
    return df_dropped

# Returns a list of the intervals contained in the dataset
# Input: Pandas Dataframe
# Output: List of strings
def get_intervals(df):
    extracted_series = df.columns.to_series().str.extract(r'(\d{1,4}_\d{2,4})')
    
    single_series = extracted_series.stack() 
    
    unique_intervals_array = single_series.dropna().unique()
    
    interval_list = list(unique_intervals_array) 

    def sort_key(interval):
        first_number_str = interval.split('_')[0]
        return int(first_number_str)

    sorted_intervals = sorted(interval_list, key=sort_key)

    print(f"Intervals found: {sorted_intervals}")
    return pd.Series(sorted_intervals)

# Creates a dataset with the specified target interval and the specified source intervals
# Input: Pandas Dataframe, List of intervals, Index of the first source interval, Index of the target interval
# Output: Pandas Dataframe
def create_dataset_from_intervals(df, intervals, start_i, drop_na=True):

    intervals_to_keep = intervals[start_i:]

    cols_to_keep = df.columns[
        df.columns.str.contains(r'Target') |
        df.columns.str.contains('|'.join(intervals_to_keep)) |
        ~(df.columns.str.contains(r'\d{1,4}_\d{2,4}', na=False))
    ]

    df = df[cols_to_keep]

    if drop_na:
        df = df.dropna()
    else:
        als_cols = df.columns[df.columns.str.startswith('ALS_')]
        als_cols.append('Target')
        df = df.dropna(subset=als_cols)

    return df

# Preprocesses combined dataset and saves to csv file
# Input: Pandas Dataframe, String path to the output csv file
# Output: Pandas Dataframe
def preprocess_combined_dataset(df, file_name, drop_na=True):
    df = drop_alsfrs_cols(df)
    df.rename(columns={'ALS_ALSFRS_R_Total_Qi+1_Central': 'Target'}, inplace=True)

    df = encode_simple_categorical(df)

    df = drop_features(df, thresh=80)

    if drop_na:
        df = df.dropna()
    else:
        als_cols = df.columns[df.columns.str.startswith('ALS_')]
        als_cols.append('Target')
        df = df.dropna(subset=als_cols)
    df.to_csv(file_name, index=False)
    print(f"Dataset {file_name}.")

    # calculate_fill_rates(df).to_csv(f'{file_name}_fill_rate.csv', index=False)

    return df

# Preprocesses combined datasets from all csv files in a folder and saves them to csv files
# Input: String path to the folder containing the csv files, String prefix for the generated csv files
# Output: Pandas Dataframe
def preprocess_combined_datasets_from_folder(folder_path, file_prefix, drop_na=True):
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

        
        output_file_name = f'{file_prefix}{csv_file}'
        print(f"Processing {csv_file}")
        df = preprocess_combined_dataset(df, output_file_name, drop_na=drop_na)

        datasets_sizes = pd.concat([datasets_sizes, pd.DataFrame({"dataset": [output_file_name], 
                                                                  "size": [len(df)],
                                                                  "nb_cols": [len(df.columns)],
                                                                  "nb_cols_non_als": [len(df.columns) - sum(
                                                                      1 for c in df.columns
                                                                      if c.startswith('ALS') or c == 'Target'
                                                                  )]
                                                                  })])
    
    return datasets_sizes

# Separate dataset with all intervals into multiple datasets of all combinaisons of consecutive intervalls and saves them to csv files outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe, String prefix for the generated csv files
# Output: Pandas Dataframe
def separate_datasets_by_intervals(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED', drop_na=True):
    intervals = get_intervals(df)

    datasets_size = pd.DataFrame({"dataset": [], 
                    "size": [],
                    "nb_cols": [],
                    "nb_cols_non_als": []})
    
    target_i = len(intervals)

    for start_i in range(target_i):
        df_temp = create_dataset_from_intervals(df, intervals, start_i, drop_na=drop_na)
        csv_name = f'{file_prefix}_{start_i*3}_{target_i*3}M.csv'
        df_temp.to_csv(csv_name, index=False)
        print(f"Dataset {csv_name} created.")

        calculate_fill_rates(df_temp).to_csv(f'{file_prefix}_fill_rate_{start_i*3}_{target_i*3}M.csv', index=False)

        datasets_size = pd.concat([
            datasets_size,
            pd.DataFrame({
                "dataset": [csv_name],
                "size": [len(df_temp)],
                "nb_cols": [len(df_temp.columns)],
                "nb_cols_non_als": [len(df_temp.columns) - sum(
                    1 for c in df_temp.columns
                    if c.startswith('ALS') or c == 'Target'
                )]
            })
        ], ignore_index=True)

    return datasets_size

# Main function to generate datasets from the original dataset and save them to csv files, also outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def generate_datasets(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED', drop_na=True):
    intervals = get_intervals(df)
    target_i = len(intervals) - 1
    df.rename(columns={f'ALS_ALSFRS_R_Total_{intervals[target_i]}_Central': 'Target'}, inplace=True)

    df = drop_alsfrs_cols(df)

    df = encode_simple_categorical(df)

    df = drop_features(df, thresh=80)

    calculate_fill_rates(df).to_csv(f'{file_prefix}_fill_rate_0_{target_i*3}M.csv', index=False)

    datasets_size = separate_datasets_by_intervals(df, file_prefix, drop_na=drop_na)

    return datasets_size

# Generates datasets from all csv files in a folder and saves them to csv files, also outputs a dataframe of the size of the generated datasets
# Input: String path to the folder containing the csv files, String prefix for the generated csv files
# Output: Pandas Dataframe
def generate_datasets_from_folder(folder_path, file_prefix, drop_na=True):
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
        datasets_size = generate_datasets(df, file_prefix, drop_na=drop_na)
        datasets_sizes.append(datasets_size)
    
    return pd.concat(datasets_sizes, ignore_index=True)

fixed_path = '../best_performing_merge/Fixed'
sliding_windows_path = '../best_performing_merge/Sliding_windows'
first_symptoms_path = '../best_performing_merge/First_symptoms'

os.makedirs(os.path.dirname("datasets/best_performing_merge/Fixed/"), exist_ok=True)
os.makedirs(os.path.dirname("datasets/best_performing_merge/Sliding_windows/"), exist_ok=True)
os.makedirs(os.path.dirname("datasets/best_performing_merge/First_symptoms/"), exist_ok=True)

# Generate datasets and save them to csv files, also outputs a dataframe of the size of the generated datasets
# datasets_size_Fixed = generate_datasets_from_folder(fixed_path, 'datasets/best_performing_merge/Fixed/', drop_na=True)
# datasets_size_Fixed.to_csv('datasets/best_performing_merge_Fixed_datasets_size.csv', index=False)

datasets_size_Sliding_Windows = preprocess_combined_datasets_from_folder(sliding_windows_path, 'datasets/best_performing_merge/Sliding_windows/', drop_na=True)
datasets_size_Sliding_Windows.to_csv('datasets/best_performing_merge_Sliding_windows_datasets_size.csv', index=False)

# datasets_size_F_S_Fixed = generate_datasets_from_folder(first_symptoms_path, 'datasets/best_performing_merge/First_symptoms/', drop_na=True)
# datasets_size_F_S_Fixed.to_csv('datasets/best_performing_merge_First_symptoms_datasets_size.csv', index=False)
