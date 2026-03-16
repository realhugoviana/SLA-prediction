import pandas as pd

runlog_data = pd.read_csv("MLP_classifier/stats_entrainement/Combined/runlog_summary.csv")

best_trials = runlog_data.loc[runlog_data.groupby("dataset")["mae"].idxmin()]

best_trials.to_csv("MLP_classifier/stats_entrainement/Combined/best_trials.csv", index=False)