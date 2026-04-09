import pandas as pd

runlog_data = pd.read_csv("MLP_regression/stats_entrainement/MLP_ALSFRS-R_COMBINED_15_03/runlog_summary.csv")

best_trials = runlog_data.loc[runlog_data.groupby("dataset")["mae"].idxmin()]

best_trials.to_csv("MLP_regression/stats_entrainement/MLP_ALSFRS-R_COMBINED_15_03/best_trials.csv", index=False)