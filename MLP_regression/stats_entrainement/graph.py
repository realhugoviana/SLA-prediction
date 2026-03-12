import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Times New Roman font
rcParams['font.family'] = 'Times New Roman'

# Load data
df = pd.read_csv("stats_entrainement/R2_MLP_RF.csv", sep=';')
df = df[['Model', 'Target', 'MAE', 'RMSE', 'R2']]

order = sorted(df["Target"].unique())
hue_order = ["MLP", "RF"]
palette = "Set1"

# Create figure with 2x2 grid, transparent background
fig, axes = plt.subplots(2, 2, figsize=(14,12), facecolor='none')
axes = axes.flatten()

# --- MAE ---
sns.barplot(ax=axes[0], data=df, x="Target", y="MAE", hue="Model", order=order, hue_order=hue_order, palette=palette)
axes[0].set_facecolor('none')
axes[0].set_title("MAE", fontsize=16)
axes[0].set_xlabel("Target", fontsize=14)
axes[0].set_ylabel("MAE", fontsize=14)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
sns.despine(ax=axes[0])

# --- RMSE ---
sns.barplot(ax=axes[1], data=df, x="Target", y="RMSE", hue="Model", order=order, hue_order=hue_order, palette=palette)
axes[1].set_facecolor('none')
axes[1].set_title("RMSE", fontsize=16)
axes[1].set_xlabel("Target", fontsize=14)
axes[1].set_ylabel("RMSE", fontsize=14)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
sns.despine(ax=axes[1])

# --- R² ---
sns.barplot(ax=axes[2], data=df, x="Target", y="R2", hue="Model", order=order, hue_order=hue_order, palette=palette)
axes[2].set_facecolor('none')
axes[2].set_title("R²", fontsize=16)
axes[2].set_xlabel("Target", fontsize=14)
axes[2].set_ylabel("R²", fontsize=14)
axes[2].tick_params(axis='x', labelsize=12)
axes[2].tick_params(axis='y', labelsize=12)
sns.despine(ax=axes[2])

# Remove the empty 4th subplot
fig.delaxes(axes[3])

# Common legend with Times New Roman font, transparent background
handles, labels = axes[2].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12, prop={'family':'Times New Roman'})
legend.get_frame().set_alpha(0)

# Remove individual legends
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()

plt.tight_layout(rect=[0,0,1,0.95])
plt.show()