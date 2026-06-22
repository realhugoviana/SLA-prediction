import pandas as pd
import numpy as np
from scipy import interpolate
import re

def get_features(df):
    temporal_cols = df.columns[~(df.columns.str.contains('observation_count')) &
                               ~(df.columns.str.contains('subject_id')) & 
                               ~(df.columns.str.contains("Delta"))]

    features = temporal_cols.str.replace(r'ALS_\d{1,2}_', '', regex=True).unique()
    features = features.str.replace(r'ALS_', '', regex=True).unique()

    return features

def make_series(df):
    df_series = df.copy()
    features = get_features(df_series)

    df_series["ALSFRS_Delta_series"] = df_series[df_series.columns[df_series.columns.str.contains("Delta")]].apply(list, axis=1)
    df_series["ALSFRS_Delta_series"] = df_series["ALSFRS_Delta_series"].apply(lambda lst: sorted([x for x in lst if not np.isnan(x)]))
    
    for feature in features:
        feature_cols = df_series.columns[df_series.columns.str.contains(feature) & 
                                         ~(df_series.columns.str.contains('pairs'))]

        def build_pairs(row):
            deltas = row["ALSFRS_Delta_series"]
            values = row[feature_cols].tolist()

            min_len = min(len(deltas), len(values))

            pairs = [
                (deltas[i], values[i])
                for i in range(len(deltas))
                if not np.isnan(values[i])
            ]

            return pairs

        df_series[f'{feature}_pairs'] = df_series.apply(build_pairs, axis=1)

    return df_series
    
def compute_spline_interpolation(x, tck):
    try:
        return interpolate.splev(x, tck)
    except Exception as e:
        print(e)
        return np.nan

def add_splines(df):
    df_splines = df.copy()

    pair_cols = [c for c in df_splines.columns if c.endswith("_pairs")]

    for col in pair_cols:
        feature = col.replace("_pairs", "")

        def build_spline(row):
            pairs = row[col]


            if len(pairs) < 2:
                return np.nan

            df_tmp = pd.DataFrame(pairs, columns=["delta", "value"])

            k = min(3, len(df_tmp) - 1)
            
            try:
                return interpolate.splrep(df_tmp['delta'], df_tmp['value'], k=k)
            except Exception as e:
                # This catches ValueError: m > k must hold and other fitting failures
                print("=========================================================================")
                print(row['subject_id'])
                print(feature)
                print(pairs)
                print(f"Error: {e}")
                return np.nan # Return NaN if the math fails


        df_splines[f"{feature}_splines"] = df_splines.apply(build_spline, axis=1)

    return df_splines



def add_missing_months(df):
    df_missing = df.copy()

    pair_cols = [c for c in df.columns if c.endswith("_pairs")]


    for col in pair_cols:
        feature = col.replace("_pairs", "")

        def compute_missing(row):
            pairs = row[col]

            if len(pairs) == 0:
                return []

            df_tmp = pd.DataFrame(pairs, columns=["delta", "value"])
            
            months = df_tmp['delta'] // 30
            months_range = np.arange(months.min(), months.max()+1)
            missing_months = sorted(list(set(months_range) - set(months)))

            return missing_months

        df_missing[f"{feature}_missing_months"] = df_missing.apply(
            lambda row: compute_missing(row),
            axis=1
        )

    return df_missing

def interpolate_missing_months_spline(df):
    df_interpolate = df.copy()

    missing_months_cols = [c for c in df.columns if c.endswith("_missing_months")]

    for col in missing_months_cols:
        feature = col.replace("_missing_months", "")

        def compute_spline_interpolation_pairs(row):
            if len(row[col]) == 0:
                return []
            
            deltas = [(x * 30) + 15 for x in row[col]]

            tck = row[f'{feature}_splines']

            interpolated_values = compute_spline_interpolation(deltas, tck)

            pairs = [
                (deltas[i], interpolated_values[i])
                for i in range(len(deltas))
            ]

            return pairs
        
        df_interpolate[f'{feature}_interpolated_pairs'] = df_interpolate.apply(
            lambda row: compute_spline_interpolation_pairs(row),
            axis=1
        )
    
    return df_interpolate

def interpolate_missing_months_linear(df):
    df_interpolate = df.copy()

    missing_months_cols = [c for c in df.columns if c.endswith("_missing_months")]

    for col in missing_months_cols:
        feature = col.replace("_missing_months", "")

        def compute_linear_interpolation_pairs(row):
            if len(row[col]) == 0:
                return []
            
            deltas_to_interpolate = [(x * 30) + 15 for x in row[col]]

            real_data = row[f'{feature}_pairs']

            if len(real_data) < 2:
                return []

            df_real_data = pd.DataFrame(real_data, columns=["delta", "value"])
            df_real_data = df_real_data.sort_values("delta")

            interpolated_values = np.interp(deltas_to_interpolate, df_real_data["delta"], df_real_data["value"])

            interpolated_pairs = list(zip(deltas_to_interpolate, interpolated_values))

            return interpolated_pairs
        
        df_interpolate[f'{feature}_interpolated_pairs'] = df_interpolate.apply(
            lambda row: compute_linear_interpolation_pairs(row),
            axis=1
        )
    
    return df_interpolate

def merge_interpolated_data(df):
    df_merge = df.copy()

    interpolated_cols = [c for c in df.columns if c.endswith("_interpolated_pairs")]

    for interpolated_col in interpolated_cols:
        real_col = interpolated_col.replace("_interpolated", "")
        feature = interpolated_col.replace("_interpolated_pairs", "")

        def merge_by_row(row):
            interpolated_data = row[interpolated_col]
            real_data = row[real_col]

            all_data = [*interpolated_data, *real_data]
            all_data = sorted(all_data, key=lambda x: x[0])

            return all_data
        
        df_merge[f'{feature}_full_pairs'] = df_merge.apply(
            lambda row: merge_by_row(row),
            axis=1
        )
    
    return df_merge

def make_months_intervals(df):
    df_intervals = df.copy()

    max_month = int(df_intervals['ALSFRS_Delta_series'].apply(max).max() // 30)
    months = np.arange(0, max_month+1)

    merge_cols = [c for c in df_intervals.columns if c.endswith("_full_pairs")]

    for month in months:
        for col in merge_cols:
            feature = col.replace('_full_pairs', '')
            df_intervals[f'{feature}_M{month}'] = np.nan

    for col in merge_cols:
        feature = col.replace('_full_pairs', '')
        def aggregate_pairs(row):
            df_tmp = pd.DataFrame(row[col], columns=["delta", "value"])
            df_tmp['month'] = (df_tmp['delta'] // 30).astype(int)

            aggregated_df = df_tmp.groupby('month')['value'].mean()

            for month, value in aggregated_df.items():
                row[f'{feature}_M{month}'] = value

            return row

        df_intervals = df_intervals.apply(
            lambda row: aggregate_pairs(row),
            axis=1
        )
    
    return df_intervals


def clean_df(df):
    df_clean = df.copy()

    cols_to_drop = df_clean.columns[df_clean.columns.str.contains('pairs') |
                                    df_clean.columns.str.contains('ALS_') |
                                    df_clean.columns.str.contains('splines') |
                                    df_clean.columns.str.contains('missing_months') |
                                    df_clean.columns.str.contains('series') |
                                    df_clean.columns.str.contains('-1') |
                                    df_clean.columns.str.contains('-2') |
                                    df_clean.columns.str.contains('-3')]
    
    df_clean = df_clean.drop(columns=cols_to_drop)

    def sort_key(col):
        match = re.match(r'^(.+)_M(\d+)$', col)
        if match:
            return (1, int(match.group(2)), match.group(1))  # pattern cols: sort by month, then feature
        return (0, 0, col)  # non-pattern cols: sort alphabetically, placed first

    df_clean = df_clean[sorted(df_clean.columns, key=sort_key)]


    return df_clean