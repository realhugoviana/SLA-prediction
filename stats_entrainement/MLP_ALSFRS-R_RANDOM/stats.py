import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_palette("pastel")

df_mae = pd.read_csv("stats_entrainement/MLP_ALSFRS-R_RANDOM/MAE.csv", sep=';')
df_r2 = pd.read_csv("stats_entrainement/MLP_ALSFRS-R_RANDOM/R2.csv", sep=';')
df_rmse = pd.read_csv("stats_entrainement/MLP_ALSFRS-R_RANDOM/RMSE.csv", sep=';')

df_mae[['dataset', 'trial']] = df_mae['Run'].str.split('/', expand=True)
df_r2[['dataset', 'trial']] = df_r2['Run'].str.split('/', expand=True)
df_rmse[['dataset', 'trial']] = df_rmse['Run'].str.split('/', expand=True)

df_mae["Value"] = df_mae["Value"].str.replace(",", ".").astype(float)
df_r2["Value"] = df_r2["Value"].str.replace(",", ".").astype(float)
df_rmse["Value"] = df_rmse["Value"].str.replace(",", ".").astype(float)

df_batch_size = pd.read_csv("stats_entrainement/MLP_ALSFRS-R_RANDOM/batch_size.csv")
df_batch_size = df_batch_size[['dataset', 'trial', 'batch_size']]
df_batch_size["dataset"] = df_batch_size["dataset"].str.replace("MLP_alsfrs-r_", "")

df_mae = df_mae.merge(df_batch_size, on=["dataset", "trial"])
df_mae["n_units"] = df_mae["n_units"].astype(str).str.replace("\u202f", "")
df_mae["n_units"] = pd.to_numeric(df_mae["n_units"], errors='coerce')

unit_order = sorted(df_mae["n_units"].unique())

# Create the boxplot
plt.figure(figsize=(8,6), facecolor='none')
ax = sns.boxplot(x="n_units", 
                 y="Value", 
                 data=df_mae, 
                 order=unit_order,
                 palette="Set2",
                 boxprops=dict(edgecolor='black'),
                 medianprops=dict(color='black'))

ax.set_facecolor('none')

# Set title and axes labels
ax.set_title("Distribution of the MAE by number of neurons per layer", fontsize=16, fontname="Times New Roman")
ax.set_xlabel("Number of neurons", fontsize=14, fontname="Times New Roman")
ax.set_ylabel("MAE", fontsize=14, fontname="Times New Roman")

# Remove the automatic 'Boxplot grouped by...' title
plt.suptitle("")

# Optional: improve style
plt.xticks(fontsize=12, fontname="Times New Roman")
plt.yticks(fontsize=12, fontname="Times New Roman")

sns.despine()
# Show the plot
plt.show()

# df_mae.boxplot(column="Value", by="criterion", grid=False)

# df_mae.boxplot(column="Value", by="n_layer", grid=False)

# df_mae.boxplot(column="Value", by="n_units", grid=False)

# df_mae.boxplot(column="Value", by="optimizer", grid=False)

# df_mae.boxplot(column="Value", by="activation", grid=False)

# df_mae.boxplot(column="Value", by="decroissant", grid=False)

# df_mae.plot.scatter(x="learning_rate", y="Value")

# df_mae.boxplot(column="Value", by="learning_rate", grid=False)

# df_mae.boxplot(column="Value", by="batch_size", grid=False)

# plt.show()

# def top5_by_dataset(df):
#     return (
#         df.sort_values(["dataset", "Value"])   
#           .groupby("dataset")
#           .head(5)                
#           .reset_index(drop=True)
#     )

# # Application aux trois métriques
# top5_mae = top5_by_dataset(df_mae)
# top5_r2 = top5_by_dataset(df_r2)
# top5_rmse = top5_by_dataset(df_rmse)

# # Affichage
# print("Top 5 MAE:")
# print(top5_mae)

# print("\nTop 5 R2:")
# print(top5_r2)

# print("\nTop 5 RMSE:")
# print(top5_rmse)

