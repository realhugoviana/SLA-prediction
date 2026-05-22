import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def stats_desc(col):

    stats = {
        "count" : col.count(),
        "moyenne" : col.mean(),
        "médiane" : col.median(),
        "écart-type" : col.std(),
        "variance" : col.var(),
        "min" : col.min(),
        "max" : col.max(),
        "étendue" : col.max() - col.min(),
        "Q1" : col.quantile(0.25),
        "Q3" : col.quantile(0.75),
        "IQR" : col.quantile(0.75) - col.quantile(0.25)
    }

    return pd.Series(stats, name=col.name)

def plot_discrete(col, trimestre):
    col = col.dropna()

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f"Distribution de Target — {trimestre}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    # --- Boxplot ---
    ax1 = fig.add_subplot(gs[0])
    ax1.boxplot(col, vert=True, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6),
                medianprops=dict(color="red", linewidth=2))
    ax1.set_title("Boxplot")
    ax1.set_ylabel("Target")
    ax1.set_xticks([])

    # --- Diagramme en bâtons ---
    ax2 = fig.add_subplot(gs[1])
    counts = col.value_counts().sort_index()
    ax2.bar(counts.index, counts.values, color="steelblue", alpha=0.6, edgecolor="white")
    ax2.set_title("Diagramme en bâtons")
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Effectif")
    ax2.set_xticks(counts.index)

    plt.tight_layout()
    plt.savefig(f"distribution_{trimestre}.png", dpi=150, bbox_inches="tight")
    plt.show()

fichiers = {"../datasets/achives_8_04_26/to_train/MLP_alsfrs-r_T1_T2.csv": "T2",
            "../datasets/achives_8_04_26/to_train/MLP_alsfrs-r_T1-T2_T3.csv": "T3",
            "../datasets/achives_8_04_26/to_train/MLP_alsfrs-r_T1-T2-T3_T4.csv": "T4",
            "../datasets/achives_8_04_26/to_train/MLP_alsfrs-r_T1-T2-T3-T4_T5.csv": "T5"}

resultats = []

for fichier, trimestre in fichiers.items():
    df = pd.read_csv(fichier)

    serie = stats_desc(df["Target"])
    serie.name = trimestre
    resultats.append(serie)

    plot_discrete(df["Target"], trimestre)

stats = pd.DataFrame(resultats)
stats.index.name = "trimestre"

stats.to_csv("stats_desc.csv")