import pandas as pd
import numpy as np
import os

# Drops questions 10 and ALSFRS scores columns (not ALSFRS-R score)
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def drop_alsfrs_cols(df):
    cols_to_drop = df.columns[
        df.columns.str.contains(
            r'ALS_ALSFRS_Total|ALS_Q10'
        )
    ]

    df = df.drop(columns=cols_to_drop)

    return df

# Returns a list of the intervals contained in the dataset
# Input: Pandas Dataframe
# Output: List of strings
def get_intervals(df):
    intervals = (
        df.columns.to_series()
            .str.extract(r'(\d{1,4}_\d{2,4})')[0]
            .dropna()
            .unique()
    )
    return intervals

# Creates a dataset with the specified target interval and the specified source intervals
# Input: Pandas Dataframe, List of intervals, Index of the first source interval, Index of the target interval
# Output: Pandas Dataframe
def create_dataset_from_intervals(df, intervals, start_i, target_i):
    df.rename(columns={f'ALS_ALSFRS_R_Total_{intervals[target_i]}_Central': 'Target'}, inplace=True)

    intervals_to_keep = intervals[start_i:target_i]

    cols_to_keep = df.columns[
        df.columns.str.contains(r'Target') |
        df.columns.str.contains('|'.join(intervals_to_keep))
    ]

    df = df[cols_to_keep]

    return df.dropna()

# Preprocesses combined dataset and saves to csv file
# Input: Pandas Dataframe, String path to the output csv file
# Output: None
def preprocess_combined_dataset(df, file_name):
    df = drop_alsfrs_cols(df)
    df.rename(columns={'ALS_ALSFRS_R_Total_Qi+1_Central': 'Target'}, inplace=True)

    df = df.dropna()
    df.to_csv(file_name, index=False)

# Preprocesses combined datasets from all csv files in a folder and saves them to csv files
# Input: String path to the folder containing the csv files, String prefix for the generated csv files
# Output: None
def preprocess_combined_datasets_from_folder(folder_path, file_prefix):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
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
        preprocess_combined_dataset(df, output_file_name)

# Separate dataset with all intervals into multiple datasets of all combinaisons of consecutive intervalls and saves them to csv files outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe, String prefix for the generated csv files
# Output: Pandas Dataframe
def separate_datasets_by_intervals(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED'):
    intervals = get_intervals(df)

    datasets_size = pd.DataFrame({"dataset": [], 
                    "size": []})
    
    target_i = len(intervals) - 1

    for start_i in range(target_i):
        df_temp = create_dataset_from_intervals(df, intervals, start_i, target_i)
        csv_name = f'{file_prefix}_{start_i*3}_{target_i*3}M.csv'
        df_temp.to_csv(csv_name, index=False)

        datasets_size = pd.concat([datasets_size, pd.DataFrame({"dataset": [csv_name], "size": [len(df_temp)]})])

    return datasets_size

# Main function to generate datasets from the original dataset and save them to csv files, also outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe
# Output: Pandas Dataframe
def generate_datasets(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED'):
    df = drop_alsfrs_cols(df)

    datasets_size = separate_datasets_by_intervals(df, file_prefix)

    return datasets_size

# Generates datasets from all csv files in a folder and saves them to csv files, also outputs a dataframe of the size of the generated datasets
# Input: String path to the folder containing the csv files, String prefix for the generated csv files
# Output: Pandas Dataframe
def generate_datasets_from_folder(folder_path, file_prefix):
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
        datasets_size = generate_datasets(df, file_prefix)
        datasets_sizes.append(datasets_size)
    
    return pd.concat(datasets_sizes, ignore_index=True)

# alsfrs_r_fixed_path = '../donnees_04_26/ALSFRS/Fixed'
alsfrs_r_combined_path = '../donnees_04_26/ALSFRS/Combined'
# alsfrs_r_first_symptoms_fixed_path = '../donnees_04_26/ALSFRS_First_Symptoms'

# # Generate datasets and save them to csv files, also outputs a dataframe of the size of the generated datasets
# datasets_size_Fixed = generate_datasets_from_folder(alsfrs_r_fixed_path, 'datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED')
preprocess_combined_datasets_from_folder(alsfrs_r_combined_path, 'datasets/ALSFRS_R_COMBINED/')
# datasets_size_Fixed.to_csv('datasets/ALSFRS_R_FIXED/datasets_size.csv', index=False)

# datasets_size_F_S_Fixed = generate_datasets_from_folder(alsfrs_r_first_symptoms_fixed_path, 'datasets/ALSFRS_R_F_S_FIXED/ALSFRS_R_F_S_FIXED')
# datasets_size_F_S_Fixed.to_csv('datasets/ALSFRS_R_F_S_FIXED/datasets_size.csv', index=False)
