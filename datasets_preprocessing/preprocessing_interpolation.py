import pandas as pd
import numpy as np
import re
import os

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

    df = df.dropna(subset=['ALSFRS_R_Total_M0', 'ALSFRS_R_Total_M1'])

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

def make_target(df):
    df = df.rename(columns={"ALSFRS_R_Total_M0": "Target"})
    df = df.drop(columns=df.columns[df.columns.str.contains(r'_M0$', regex=True)])

    return df


def sliding_windows_rnn(df):
    feature_names = df.columns[~(df.columns.str.contains('subject_id'))].str.replace(r'_M\-?\d+$', '', regex=True).unique()
    print(feature_names)

    def make_windows(row):
        sequences = {col: [] for col in df.columns}
        for col in df.columns:
            sequences[col].append(row[col])

        len_sequence = 16
        for month in range(-15, 1, 1):
            if np.isnan(row[f'ALSFRS_R_Total_M{month}']):
                len_sequence = -month

        while len_sequence > 2:
            sequences['subject_id'].append(row['subject_id'])

            for month in range(-16, 0, 1):
                for feature in feature_names:
                    if month == -16:
                        sequences[f'{feature}_M{month+1}'].append(np.nan)
                    else:
                        sequences[f'{feature}_M{month+1}'].append(sequences[f'{feature}_M{month}'][-2])
            len_sequence -= 1
        sequence_df = pd.DataFrame({**{k: v for k, v in sequences.items()}})
        print(sequence_df)
        return sequence_df

    sliding_windows_df = pd.concat(df.apply(make_windows, axis=1).tolist(), ignore_index=True)

    return sliding_windows_df

if __name__ == '__main__':
    df = pd.read_csv('../../data/PROACT_INTERPOLATION.csv')

    df = drop_unused_cols(df)

    df_rnn = align_patients_rnn(df)
    df_sliding_rnn = sliding_windows_rnn(df_rnn)
    # df_rnn = make_target(df_rnn)
    # df_sliding_rnn = make_target(df_sliding_rnn)
    df_rnn = df_rnn.drop(columns='subject_id')
    df_sliding_rnn = df_sliding_rnn.drop(columns='subject_id')
    df_rnn = df_rnn.fillna(0.0)
    df_sliding_rnn = df_sliding_rnn.fillna(0.0)

    os.makedirs('datasets/interpolation', exist_ok=True)
    df_rnn.to_csv("datasets/interpolation/fixed.csv", index=False)
    df_sliding_rnn.to_csv("datasets/interpolation/sliding_windows.csv", index=False)