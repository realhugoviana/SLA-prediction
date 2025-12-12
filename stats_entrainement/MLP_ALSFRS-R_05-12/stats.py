import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


runlog_data = pd.read_csv("stats_entrainement/MLP_ALSFRS-R_05-12/runlog_summary.csv")
runlog_data["dataset"] = runlog_data["dataset"].str.replace("MLP_alsfrs-r_", "")

best_trials = runlog_data.loc[runlog_data.groupby("dataset")["mae"].idxmin()]
best_trials = best_trials[["dataset", "n_layer", "n_units", "learning_rate", "decroissant", "mae", "rmse", "r2", "trial"]]

best_trials.to_csv("stats_entrainement/MLP_ALSFRS-R_05-12/best_trials_mae.csv", index=False)

# ########################################################################
# ###### boxplot mae by groupByHyperparams ordered by median mae #########
# ########################################################################

# # Compute max per group for ordering
# median_values = runlog_data.groupby("dataset")["mae"].median()
# ordered_groups = median_values.sort_values().index  # ascending; use ascending=False for descending

# # Create horizontal boxplot with seaborn
# sns.boxplot(
#     data=runlog_data,
#     x="mae",
#     y="dataset",
#     order=ordered_groups
# )

# plt.xlabel("mae")
# plt.ylabel("dataset")
# plt.tight_layout()
# plt.show()

# ########################################################################
# ###### barplot metrics best model #######################
# ########################################################################

# # 2. Select the columns we care about
# plot_df = best_trials.copy()

# # 3. Create a combined label with dataset and trial
# plot_df["dataset_label"] = plot_df["dataset"] + " (" + plot_df["learning_rate"].astype(str) + ")"

# # 4. Melt to long format
# long_df = plot_df.melt(
#     id_vars="dataset_label",
#     value_vars=["mae", "rmse", "r2"],
#     var_name="metric",
#     value_name="value"
# )

# # 5. Order by MAE
# ordered_labels = plot_df.sort_values("mae")["dataset_label"]

# # 6. Plot
# plt.figure(figsize=(12,6))
# sns.barplot(
#     data=long_df,
#     x="dataset_label",
#     y="value",
#     hue="metric",
#     order=ordered_labels,
#     palette="pastel"
# )

# plt.ylabel("Value")
# plt.xlabel("Dataset (learning_rate)")
# plt.xticks(rotation=45, ha="right")
# plt.title("MAE, RMSE, R² for best MAE")
# plt.legend(title="Metric")
# plt.tight_layout()
# plt.show()


# ######################################################
# ########## Scatterplot of mae by learning rate########
# ######################################################

# plt.figure(figsize=(10,6))
# sns.scatterplot(
#     data=runlog_data,
#     x="learning_rate",
#     y="mae",
#     s=80                      # taille des points
# )

# plt.xscale("log")
# plt.xlabel("Learning rate")
# plt.ylabel("MAE")
# plt.title("MAE en fonction du learning rate")
# plt.tight_layout()
# plt.show()
