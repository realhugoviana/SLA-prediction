import pandas as pd
import interpolation_utils as ut

df = pd.read_csv('../../data/PRO-ACT_V8/PROACT_ALSFRS_v8.csv')

df_series = ut.make_series(df)

df_months = ut.add_months(df_series)

df_splines = ut.add_splines(df_months)

print(df_splines.head())