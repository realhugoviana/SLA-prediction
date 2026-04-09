import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# =========================
# GLOBAL FONT CONFIGURATION
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,  # slightly larger ticks
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black"
})

# Use ticks style to remove background grid lines
sns.set_style("ticks")
sns.set_context("talk")

# =========================
# DATA LOADING
# =========================
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

df_mae["n_units"] = (
    df_mae["n_units"]
    .astype(str)
    .str.replace("\u202f", "")
)
df_mae["n_units"] = pd.to_numeric(df_mae["n_units"], errors='coerce')

unit_order = sorted(df_mae["n_units"].unique())

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 6), facecolor='none')

ax = sns.boxplot(
    x="n_units",
    y="Value",
    data=df_mae,
    order=unit_order,
    color="#FF968D",  # Only fill color
    boxprops=dict(edgecolor="black", linewidth=4),
    whiskerprops=dict(color="black", linewidth=6),
    capprops=dict(color="black", linewidth=6),
    medianprops=dict(color="black", linewidth=6)
)

ax.set_facecolor('none')

# ---- BLACK AXES ----
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(2.5)

ax.tick_params(axis="both", colors="black", width=2.0, direction="out")

# ---- TITLE AND LABELS (LARGER) ----
# ax.set_title(
#     "Distribution of the MAE by number of neurons per layer",
#     fontsize=100,  # increased
#     color="black"
# )
ax.set_xlabel("Number of neurons", fontsize=50, color="black")  # increased
ax.set_ylabel("MAE", fontsize=50, color="black")  # increased

plt.suptitle("")
sns.despine(offset=0, trim=False)  # keep axes lines

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

