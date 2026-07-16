import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from processing_utils import encode_simple_categorical, drop_unused_cols, get_intervals
import os

def get_als_features(df):
    columns = df.columns[df.columns.str.startswith('ALS_', na=False)]
    return list(sorted(columns.str.replace(r'(\d{1,4}_\d{2,4})', '', regex=True).unique()))

def mice(df, thresh):
    df_thresh = df.copy()
    df_thresh = df_thresh.dropna(thresh=(thresh / 100) * len(df_thresh), axis=1)

    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42, max_iter=10)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_thresh), columns=df_thresh.columns)

    return df_imputed

def locf(df):
    intervals = get_intervals(df)
    features = get_als_features(df)

    df_locf = df.copy()

    for feature in features:
        for i in range(len(intervals)-1):
            interval = intervals[i+1]
            previous_interval = intervals[i]
            df_locf[f'{feature}_{interval}'] = df_locf[f'{feature}_{interval}'].fillna(df_locf[f'{feature}_{previous_interval}'])

    return df_locf

def mice_locf(df, thresh):
    df_locf = locf(df)
    df_mice_locf = mice(df_locf, thresh)
    return df_mice_locf

if __name__ == "__main__":
    df = pd.read_csv("../../data/donnees_04_26/ALL_DATA_UNION/Fixed/to_train/Fixed_0_12M.csv")
    df = encode_simple_categorical(df)
    df = drop_unused_cols(df)

    os.makedirs("datasets/MICE_LOCF", exist_ok=True)

    for thresh in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print(f"Processing threshold: {thresh}%")
        df_mice = mice(df, thresh)
        print(f"Completed MICE imputation for threshold: {thresh}%")
        df_mice_locf = mice_locf(df, thresh)
        print(f"Completed MICE-LOCF imputation for threshold: {thresh}%")

        df_mice.to_csv(f"datasets/MICE_LOCF/ALL_DATA_UNION_Fixed_0_12M_mice_{thresh}.csv", index=False)
        df_mice_locf.to_csv(f"datasets/MICE_LOCF/ALL_DATA_UNION_Fixed_0_12M_mice_locf_{thresh}.csv", index=False)