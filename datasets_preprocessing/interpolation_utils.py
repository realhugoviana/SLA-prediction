import pandas as pd
import numpy as np
from scipy import interpolate

def get_features(df):
    temporal_cols = df.columns[~(df.columns.str.contains('observation_count')) &
                               ~(df.columns.str.contains('subject_id'))]

    features = temporal_cols.str.replace(r'ALS_\d{1,2}_', '', regex=True).unique()
    features = features.str.replace(r'ALS_', '', regex=True).unique()

    return features

def make_series(df):
    df_series = df.copy()
    features = get_features(df_series)

    for feature in features:
        feature_cols = df_series.columns[df_series.columns.str.contains(feature)]

        df_series[f'{feature}_series'] = df_series[feature_cols].apply(list, axis=1)
        df_series[f'{feature}_series'] = df_series[f'{feature}_series'].apply(lambda lst: [x for x in lst if not np.isnan(x)])

    return df_series

def add_months(df):
    df_months = df.copy()

    df_months['Months_series'] = df_months['ALSFRS_Delta_series'].apply(lambda lst: [x // 30 for x in lst])

    return df_months

def add_splines(df):
    df_splines = df.copy()
    features = get_features(df)

    for feature in features:
        df_splines[f'{feature}_splines'] = df_splines.apply(
            lambda row: interpolate.splrep(row['ALSFRS_Delta_series'], row[f'{feature}_series']),
            axis=1
        )

    return df_splines
    
    


