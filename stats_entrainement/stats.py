import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_mae = pd.read_csv("stats_entrainement/MAE.csv", sep=';')
df_r2 = pd.read_csv("stats_entrainement/R2.csv", sep=';')
df_rmse = pd.read_csv("stats_entrainement/RMSE.csv", sep=';')

df_mae[['dataset', 'trial']] = df_mae['Run'].str.split('/', expand=True)
df_mae.drop(columns=['Run'], inplace=True)
df_r2[['dataset', 'trial']] = df_r2['Run'].str.split('/', expand=True)
df_r2.drop(columns=['Run'], inplace=True)
df_rmse[['dataset', 'trial']] = df_rmse['Run'].str.split('/', expand=True)
df_rmse.drop(columns=['Run'], inplace=True)


print(df_mae.head())