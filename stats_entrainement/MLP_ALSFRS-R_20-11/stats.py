import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


runlog_data = pd.read_csv("stats_entrainement/MLP_ALSFRS-R_20-11/runlog_summary.csv")
runlog_data["dataset"] = runlog_data["dataset"].str.replace("MLP_alsfrs-r_", "")
runlog_data["groupByHyperparams"] = runlog_data["n_layer"].astype(str) + " - " + \
                                     runlog_data["n_units"].astype(str) + " - " + \
                                     runlog_data["learning_rate"].astype(str) + " - " + \
                                     runlog_data["decroissant"].astype(str)

##################################################
###### Métriques centrées réduites ###############
##################################################

mae_avg = runlog_data["mae"].mean()
mae_std = runlog_data["mae"].std()
runlog_data["mae_zscore"] = (runlog_data["mae"] - mae_avg) / mae_std

rmse_avg = runlog_data["rmse"].mean()
rmse_std = runlog_data["rmse"].std()
runlog_data["rmse_zscore"] = (runlog_data["rmse"] - rmse_avg) / rmse_std

runlog_data["r2_minimise"] = 1 - runlog_data["r2"]
r2_avg = runlog_data["r2_minimise"].mean()
r2_std = runlog_data["r2_minimise"].std()
runlog_data["r2_zscore"] = (runlog_data["r2_minimise"] - r2_avg) / r2_std

runlog_data["combined_zscore"] = (runlog_data["mae_zscore"] +
                                  runlog_data["rmse_zscore"] +
                                  runlog_data["r2_zscore"])

best_trials = runlog_data.loc[runlog_data.groupby("dataset")["mae"].idxmin()]
best_trials = best_trials[["dataset", "n_layer", "n_units", "decroissant", "learning_rate", "mae", "rmse", "r2", "trial"]]

best_trials.to_csv("stats_entrainement/MLP_ALSFRS-R_20-11/best_trials_mae.csv", index=False)


########################################################################
###### boxplot mae by groupByHyperparams ordered by median mae #########
########################################################################

# # Compute max per group for ordering
# median_values = runlog_data.groupby("groupByHyperparams")["mae"].median()
# ordered_groups = median_values.sort_values().index  # ascending; use ascending=False for descending

# # Create horizontal boxplot with seaborn
# sns.boxplot(
#     data=runlog_data,
#     x="mae",
#     y="groupByHyperparams",
#     order=ordered_groups
# )

# plt.xlabel("mae")
# plt.ylabel("groupByHyperparams (sorted by max)")
# plt.tight_layout()
# plt.show()

########################################################################
###### barplot metrics best model #######################
########################################################################

# 2. Select the columns we care about
plot_df = best_trials[["dataset", "trial", "mae", "rmse", "r2", "combined_zscore"]].copy()

# 3. Create a combined label with dataset and trial
plot_df["dataset_label"] = plot_df["dataset"] + " (" + plot_df["trial"].astype(str) + ")"

# 4. Melt to long format
long_df = plot_df.melt(
    id_vars="dataset_label",
    value_vars=["mae", "rmse", "r2"],
    var_name="metric",
    value_name="value"
)

# 5. Order by MAE
ordered_labels = plot_df.sort_values("combined_zscore")["dataset_label"]

# 6. Plot
plt.figure(figsize=(12,6))
sns.barplot(
    data=long_df,
    x="dataset_label",
    y="value",
    hue="metric",
    order=ordered_labels,
    palette="pastel"
)

plt.ylabel("Value")
plt.xlabel("Dataset (trial)")
plt.xticks(rotation=45, ha="right")
plt.title("MAE, RMSE, R² for best model per dataset")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


