import pandas as pd
import os

csv_file = "../../data/data_as_time_series_120m_data_Only_ALSFRS.csv"

df = pd.read_csv(csv_file)

df_15M = df.iloc[:, :18]
df_15M.columns = ['subject_id', 'feature', '0_30', '30_60', '60_90', '90_120', '120_150', '150_180', '180_210', '210_240', '240_270', '270_300', '300_330', '330_360', '360_390', '390_420', '420_450', '450_480']

df_indexed = df_15M.set_index(['subject_id', 'feature'])
stacked_series = df_indexed.stack() 
stacked_series.name = 'value'
df_long = stacked_series.reset_index()

final_df = df_long.pivot_table(
    index='subject_id',
    columns=['feature', 'level_2'],
    values='value',
    aggfunc='first'
)
final_df.columns = [f'{col1}_{col2}' for col1, col2 in final_df.columns]

final_df_15M = final_df.copy()
final_df_15M = final_df_15M.rename(columns={"ALSFRS_Total_450_480": "Target"})

final_df_15M = final_df_15M[final_df_15M.columns[~final_df_15M.columns.str.contains('_450_480')]]

final_df_3M = final_df.copy()
final_df_3M = final_df_3M.rename(columns={"ALSFRS_Total_90_120": "Target"})

final_df_3M = final_df_3M[final_df_3M.columns[final_df_3M.columns.str.contains('_0_30') |
                                              final_df_3M.columns.str.contains('_30_60') |
                                              final_df_3M.columns.str.contains('_60_90') |
                                              final_df_3M.columns.str.contains('Target')]]

final_df_6M = final_df.copy()

final_df_6M = final_df_6M.rename(columns={"ALSFRS_Total_180_210": "Target"})

final_df_6M = final_df_6M[final_df_6M.columns[final_df_6M.columns.str.contains('_0_30') |
                                              final_df_6M.columns.str.contains('_30_60') |
                                              final_df_6M.columns.str.contains('_60_90') |
                                              final_df_6M.columns.str.contains('_90_120') |
                                              final_df_6M.columns.str.contains('_120_150') |
                                              final_df_6M.columns.str.contains('_150_180') |
                                              final_df_6M.columns.str.contains('Target')]]

final_df_9M = final_df.copy()

final_df_9M = final_df_9M.rename(columns={"ALSFRS_Total_270_300": "Target"})

final_df_9M = final_df_9M[final_df_9M.columns[final_df_9M.columns.str.contains('_0_30') |
                                              final_df_9M.columns.str.contains('_30_60') |
                                              final_df_9M.columns.str.contains('_60_90') |
                                              final_df_9M.columns.str.contains('_90_120') |
                                              final_df_9M.columns.str.contains('_120_150') |
                                              final_df_9M.columns.str.contains('_150_180') |
                                              final_df_9M.columns.str.contains('_180_210') |
                                              final_df_9M.columns.str.contains('_210_240') |
                                              final_df_9M.columns.str.contains('_240_270') |
                                              final_df_9M.columns.str.contains('Target')]]

final_df_12M = final_df.copy()

final_df_12M = final_df_12M.rename(columns={"ALSFRS_Total_360_390": "Target"})

final_df_12M = final_df_12M[final_df_12M.columns[final_df_12M.columns.str.contains('_0_30') |
                                              final_df_12M.columns.str.contains('_30_60') |
                                              final_df_12M.columns.str.contains('_60_90') |
                                              final_df_12M.columns.str.contains('_90_120') |
                                              final_df_12M.columns.str.contains('_120_150') |
                                              final_df_12M.columns.str.contains('_150_180') |
                                              final_df_12M.columns.str.contains('_180_210') |
                                              final_df_12M.columns.str.contains('_210_240') |
                                              final_df_12M.columns.str.contains('_240_270') |
                                              final_df_12M.columns.str.contains('_270_300') |
                                              final_df_12M.columns.str.contains('_300_330') |
                                              final_df_12M.columns.str.contains('_330_360') |
                                              final_df_12M.columns.str.contains('Target')]]

os.makedirs('datasets/papaiz/', exist_ok=True)
# final_df_15M.to_csv('datasets/papaiz/papaiz_15M.csv', index=False)
# final_df_3M.to_csv('datasets/papaiz/papaiz_3M.csv', index=False)
final_df_6M.to_csv('datasets/papaiz/papaiz_6M.csv', index=False)
final_df_9M.to_csv('datasets/papaiz/papaiz_9M.csv', index=False)
final_df_12M.to_csv('datasets/papaiz/papaiz_12M.csv', index=False)