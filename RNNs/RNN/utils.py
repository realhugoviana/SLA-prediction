import pandas as pd
import numpy as np

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

def get_features(df):
    columns = df.columns
    
    return list(sorted(columns.str.replace(r'_\d{1,4}_\d{2,4}', '', regex=True).unique()))

def sort_df(df, intervals):
    sorted_columns = []

    for interval in intervals:
        interval_columns = df.columns[df.columns.str.contains(rf'_{interval}', na=False)].tolist()
        sorted_columns.extend(sorted(interval_columns))

    sorted_columns.append("Target")
    return df[sorted_columns]

def get_input_size(df):
    return len(get_features(df)) -1