import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_mae = pd.read_csv("stats_entrainement/MAE.csv", sep=';')
df_r2 = pd.read_csv("stats_entrainement/R2.csv", sep=';')
df_rmse = pd.read_csv("stats_entrainement/RMSE.csv", sep=';')

df_mae[['dataset', 'trial']] = df_mae['Run'].str.split('/', expand=True)
df_r2[['dataset', 'trial']] = df_r2['Run'].str.split('/', expand=True)
df_rmse[['dataset', 'trial']] = df_rmse['Run'].str.split('/', expand=True)

df_mae["Value"] = df_mae["Value"].str.replace(",", ".").astype(float)
df_r2["Value"] = df_r2["Value"].str.replace(",", ".").astype(float)
df_rmse["Value"] = df_rmse["Value"].str.replace(",", ".").astype(float)

df_batch_size = pd.read_csv("stats_entrainement/batch_size.csv")
df_batch_size = df_batch_size[['dataset', 'trial', 'batch_size']]
df_batch_size["dataset"] = df_batch_size["dataset"].str.replace("MLP_alsfrs-r_", "")

df_mae = df_mae.merge(df_batch_size, on=["dataset", "trial"])

df_mae.boxplot(column="Value", by="criterion", grid=False)

df_mae.boxplot(column="Value", by="n_layer", grid=False)

df_mae.boxplot(column="Value", by="n_units", grid=False)

df_mae.boxplot(column="Value", by="optimizer", grid=False)

df_mae.boxplot(column="Value", by="activation", grid=False)

df_mae.boxplot(column="Value", by="decroissant", grid=False)

df_mae.plot.scatter(x="learning_rate", y="Value")

df_mae.boxplot(column="Value", by="learning_rate", grid=False)

df_mae.boxplot(column="Value", by="batch_size", grid=False)

plt.show()

def top5_by_dataset(df):
    return (
        df.sort_values(["dataset", "Value"])   
          .groupby("dataset")
          .head(5)                
          .reset_index(drop=True)
    )

# Application aux trois métriques
top5_mae = top5_by_dataset(df_mae)
top5_r2 = top5_by_dataset(df_r2)
top5_rmse = top5_by_dataset(df_rmse)

# Affichage
print("Top 5 MAE:")
print(top5_mae)

print("\nTop 5 R2:")
print(top5_r2)

print("\nTop 5 RMSE:")
print(top5_rmse)

