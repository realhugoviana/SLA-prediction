import pandas as pd
import numpy as np

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

# Separate dataset with all intervals into multiple datasets of all combinaisons of consecutive intervalls and saves them to csv files outputs a dataframe of the size of the generated datasets
# Input: Pandas Dataframe, String prefix for the generated csv files
# Output: Pandas Dataframe
def separate_datasets_by_intervals(df, file_prefix='datasets/ALSFRS_R_FIXED/ALSFRS_R_FIXED'):
    intervals = get_intervals(df)

    datasets_size = pd.DataFrame({"dataset": [], 
                    "size": []})
    
    for target_i in range(1, len(intervals)):
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

# Unseparated datasets
df_alsfrs_r_Fixed_0_69M = pd.read_csv('../donnees_04_26/ALSFRS/Fixed_0_69M.csv')
df_alsfrs_r_F_S_Fixed_0_300M = pd.read_csv('../donnees_04_26/ALSFRS_First_Symptoms/Fixed_0_300M.csv')

# Generate datasets and save them to csv files, also outputs a dataframe of the size of the generated datasets
datasets_size_Fixed_0_69M = generate_datasets(df_alsfrs_r_Fixed_0_69M)
datasets_size_Fixed_0_69M.to_csv('datasets/ALSFRS_R_FIXED/datasets_size.csv', index=False)

datasets_size_F_S_Fixed_0_300M = generate_datasets(df_alsfrs_r_F_S_Fixed_0_300M, 'datasets/ALSFRS_R_F_S_FIXED/ALSFRS_R_F_S_FIXED')
datasets_size_F_S_Fixed_0_300M.to_csv('datasets/ALSFRS_R_F_S_FIXED/datasets_size.csv', index=False)
