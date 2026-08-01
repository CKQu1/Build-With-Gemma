from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_time_series_scores(
    values: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: str | Path,
    labels: np.ndarray | None = None,
    feature_name: str = "sensor_0",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timesteps = np.arange(len(scores))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(timesteps, values[: len(scores), 0], color="#1f77b4", linewidth=1.0, label=feature_name)
    flagged = np.isfinite(scores) & (scores > threshold)
    axes[0].scatter(timesteps[flagged], values[: len(scores), 0][flagged], color="#d62728", s=14, label="detected")
    if labels is not None:
        labeled = labels[: len(scores)].astype(bool)
        axes[0].scatter(timesteps[labeled], values[: len(scores), 0][labeled], color="#2ca02c", s=8, alpha=0.4, label="label")
    axes[0].set_ylabel("sensor value")
    axes[0].legend(loc="upper right")

    axes[1].plot(timesteps, scores, color="#444444", linewidth=1.0)
    axes[1].axhline(threshold, color="#d62728", linestyle="--", linewidth=1.0, label="threshold")
    axes[1].set_ylabel("anomaly score")
    axes[1].set_xlabel("timestep")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_distribution(scores: np.ndarray, threshold: float, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_scores = scores[np.isfinite(scores)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(valid_scores, bins=60, kde=True, color="#4c78a8", ax=ax)
    ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

