"""Regenerate the requested statistical plots from submitted and supplemental data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from reviewer_checks import build_dataset

HERE = Path(__file__).parent
SOURCE = HERE.parent / "results" / "original"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

raw, y = build_dataset()
train, _ = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=7)
pcs = PCA(n_components=4).fit_transform(StandardScaler().fit_transform(raw[train]))
fig, ax = plt.subplots(figsize=(6.2, 4.3), layout="constrained")
for label, name, marker, color in [(0, "Tone (class 0)", "o", "#245c93"),
                                  (1, "Chirp (class 1)", "^", "#ad4b13")]:
    mask = y[train] == label
    ax.scatter(pcs[mask, 0], pcs[mask, 1], label=name, marker=marker, c=color, alpha=.65, s=27)
ax.set(xlabel="PC1 (56.87% of variance)", ylabel="PC2 (24.93% of variance)")
ax.legend(frameon=False)
fig.savefig(HERE/"figure2_pca_labeled.png", dpi=300)
plt.close(fig)

metrics = pd.read_csv(HERE/"classical_controls.csv")
fig, ax = plt.subplots(figsize=(7.4, 4.7), layout="constrained")
pivot = metrics.pivot(index="seed", columns="model", values="fresh_accuracy")
controls = ["Logistic", "Traceless quadratic", "MLP 4-2-1", "General quadratic",
            "QDA", "RBF-SVM", "Random forest", "Small MLP"]
for i, model in enumerate(controls):
    values = 100*(pivot["PNN"]-pivot[model])
    ax.scatter(values, np.full(len(values), i), color="#245c93", alpha=.5, s=30)
    ax.scatter(values.mean(), i, color="#ad4b13", marker="D", s=35, zorder=3)
ax.axvline(0, color=".45", linestyle="--")
ax.set(yticks=range(len(controls)), yticklabels=controls,
       xlabel="PNN minus baseline on fresh tests (percentage points)")
ax.invert_yaxis()
fig.savefig(HERE/"figure3_paired_differences.png", dpi=300)
plt.close(fig)

noise = pd.read_csv(SOURCE/"pnn_raw_noise_accuracies.csv")
fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="constrained")
rng = np.random.default_rng(7)
summary = noise.groupby("sigma_rad")["accuracy"]
for k, values in summary:
    ax.scatter(k+rng.uniform(-.0027, .0027, len(values)), values, s=15,
               alpha=.25, color="#245c93")
levels = summary.mean().index
ax.plot(levels, summary.mean(), "D-", color="#ad4b13", lw=1.5, ms=5, label="Mean")
ax.vlines(levels, summary.quantile(.05), summary.quantile(.95), color="#245c93",
          lw=2, label="5th–95th percentile")
ax.axhline(.87, color=".5", ls="--", label="Unperturbed accuracy")
ax.set(xlabel="RMS mesh parameter perturbation (rad)", ylabel="Test accuracy",
       xticks=levels, xticklabels=[f"{x:.2f}" for x in levels], ylim=(.41, .92))
ax.tick_params(axis="x", labelrotation=55)
ax.legend(frameon=False, loc="lower left")
fig.savefig(HERE/"figure6_perturbation_spread.png", dpi=300)

shot = pd.read_csv(SOURCE/"raw_shot_accuracies.csv")
fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="constrained")
for (name, group), color, marker in zip(shot.groupby("Model"), ["#245c93", "#ad4b13"], ["o", "^"]):
    stats = group.groupby("Shots")["Accuracy"].agg(["mean", "std"])
    ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker=marker,
                color=color, capsize=3, label=name)
ax.set_xscale("log", base=2)
levels = sorted(shot["Shots"].unique())
ax.set(xlabel="Inference shots", ylabel="Accuracy on 100 held-out signals",
       xticks=levels, xticklabels=[str(x) for x in levels], ylim=(.54, .67))
ax.legend(frameon=False)
fig.savefig(HERE/"figure4_shot_sweep.png", dpi=300)
plt.close(fig)
