import pandas as pd
import interpolation_utils as ut

df_v8 = pd.read_csv('../../data/PRO-ACT_V8/PROACT_ALSFRS_v8.csv')
print(ut.get_features(df_v8))

df_pairs = ut.make_series(df_v8)

df_splines = ut.add_splines(df_pairs)

df_missing = ut.add_missing_months(df_splines)

df_interpolation = ut.interpolate_missing_months(df_missing)

df_merge = ut.merge_interpolated_data(df_interpolation)

df_intervals = ut.make_months_intervals(df_merge)

print(df_intervals)