import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

def drop_unused_cols(df):
    df = df.drop(columns=df.columns[df.columns.str.contains('ALSFRS_Total') |
                                    df.columns.str.contains('Q10')])

    cols_to_drop = []
    for col in df.columns:
        match = re.search(r'(\d+)$', str(col))

        if match:
            mois = int(match.group(1))

            if mois > 15:
                cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)

    df = df.dropna(subset=['ALSFRS_R_Total_M0', 'ALSFRS_R_Total_M1', 'ALSFRS_R_Total_M2', 'ALSFRS_R_Total_M3'])

    return df

def align_patients_rnn(df):
    feature_names = df.columns[~(df.columns.str.contains('subject_id'))].str.replace(r'_M(\d+)$', '', regex=True).unique()

    new_columns = ['subject_id']
    for t in range(-15, 1, 1):
        for feature in feature_names:
            new_columns.append(f'{feature}_M{t}')

    aligned_df = pd.DataFrame(index=df.index, columns=new_columns)
    aligned_df['subject_id'] = df['subject_id']
    for t in range(0, -16, -1):
        for feature in feature_names:
            def align_row(row):
                len_sequence = 16

                for month in reversed(range(16)):
                    if np.isnan(row[f'ALSFRS_R_Total_M{month}']):
                        len_sequence = month

                if len_sequence - 1 + t < 0:
                    return np.nan
                
                return row[f'{feature}_M{len_sequence - 1 + t}']

            aligned_df[f'{feature}_M{t}'] = df.apply(align_row, axis=1)

    return aligned_df

def split_train_test_opt(df, test_size=0.2, random_state=42):
    train_opt_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    opt_size = (len(test_df)/2)/len(train_opt_df)

    train_df, opt_df = train_test_split(train_opt_df, test_size=opt_size, random_state=random_state)

    return train_df, test_df, opt_df

def sliding_windows_rnn(df):
    feature_names = df.columns[~(df.columns.str.contains('subject_id'))].str.replace(r'_M\-?\d+$', '', regex=True).unique()

    def make_windows(row):
        sequences = {col: [] for col in df.columns}
        for col in df.columns:
            sequences[col].append(row[col])

        len_sequence = 16
        for month in range(-15, 1, 1):
            if np.isnan(row[f'ALSFRS_R_Total_M{month}']):
                len_sequence = -month

        while len_sequence > 4:
            sequences['subject_id'].append(row['subject_id'])

            for month in range(-16, 0, 1):
                for feature in feature_names:
                    if month == -16:
                        sequences[f'{feature}_M{month+1}'].append(np.nan)
                    else:
                        sequences[f'{feature}_M{month+1}'].append(sequences[f'{feature}_M{month}'][-2])
            len_sequence -= 1
        sequence_df = pd.DataFrame({**{k: v for k, v in sequences.items()}})
        return sequence_df

    sliding_windows_df = pd.concat(df.apply(make_windows, axis=1).tolist(), ignore_index=True)

    sliding_windows_df = sliding_windows_df[sliding_windows_df.columns[sliding_windows_df.columns.str.contains(rf'_M-3$') |
                                                                       sliding_windows_df.columns.str.contains(rf'_M-2$') |
                                                                       sliding_windows_df.columns.str.contains(rf'_M-1$') |
                                                                       sliding_windows_df.columns.str.contains(rf'_M0$') |
                                                                       sliding_windows_df.columns.str.contains('subject_id')]]

    return sliding_windows_df


def split_by_size(df):
    df_test = df

    dataframes = dict()
    for t in range(13, 0, -1):
        dataframes[f'df_{t}'] = df_test[df_test[f'ALSFRS_R_Total_M{-t-2}'].notna()].copy()
        df_test = df_test[df_test[f'ALSFRS_R_Total_M{-t-2}'].isna()]
        df_test = df_test[df_test.columns[~(df_test.columns.str.contains(rf'_M{-t-2}$'))]]
    
    return dataframes

def drop_all_but_baseline(df):

    cols_to_keep = df.columns[df.columns.str.contains('ALSFRS_Total') |
                              df.columns.str.contains('Target')]

    df_baseline = df[cols_to_keep]

    return df_baseline

if __name__ == '__main__':
    os.makedirs('datasets/fixed_length_interpolation/test/', exist_ok=True)
    os.makedirs('datasets/fixed_length_interpolation/optimisation/', exist_ok=True)

    df = pd.read_csv('../../data/PROACT_INTERPOLATION.csv')

    df = drop_unused_cols(df)

    print(len(df))

    df_rnn = align_patients_rnn(df)

    df_rnn_train, df_rnn_test, df_rnn_opt = split_train_test_opt(df_rnn)

    # df_sliding_rnn = sliding_windows_rnn(df_rnn_train)

    df_rnn_test = df_rnn_test.drop(columns='subject_id')
    # df_sliding_rnn = df_sliding_rnn.drop(columns='subject_id')
    df_rnn_opt = df_rnn_opt.drop(columns='subject_id')

    dfs_rnn_test = split_by_size(df_rnn_test)

    for name, df_test in dfs_rnn_test.items():
        df_test.to_csv(f'datasets/fixed_length_interpolation/test/{name}.csv', index=False)

    dfs_rnn_opt = split_by_size(df_rnn_opt)

    for name, df_opt in dfs_rnn_opt.items():
        df_opt.to_csv(f'datasets/fixed_length_interpolation/optimisation/{name}.csv', index=False)

    # df_sliding_rnn.to_csv("datasets/fixed_length_interpolation/sliding_windows.csv", index=False)