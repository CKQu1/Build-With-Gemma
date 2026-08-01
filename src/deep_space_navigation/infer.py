from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import config_from_dict
from .data import aggregate_window_errors, create_dataloaders, create_datasets, prepare_data
from .models import build_model
from .training_utils import ensure_dir, select_device
from .visualize import plot_score_distribution, plot_time_series_scores


def run_inference(checkpoint_path: str | Path, make_plots: bool = False) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = config_from_dict(checkpoint["config"])
    output_dir = ensure_dir(Path(config.output.root_dir) / config.output.experiment_name)

    bundle = prepare_data(config.data, scaler_state=checkpoint["scaler"])
    datasets = create_datasets(config.data, bundle)
    loaders = create_dataloaders(datasets, config.training)
    device = select_device(config.training.device)
    model = build_model(
        config.model,
        n_features=int(checkpoint["n_features"]),
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_scores, _ = _score_dataset(model, loaders["val"], datasets["val"], len(bundle.val), config.data.sequence_length, device)
    test_scores, counts = _score_dataset(
        model,
        loaders["test"],
        datasets["test"],
        len(bundle.test),
        config.data.sequence_length,
        device,
    )
    valid_val_scores = val_scores[np.isfinite(val_scores)]
    if len(valid_val_scores) == 0:
        raise ValueError("Validation scores are empty; cannot estimate an anomaly threshold.")
    threshold = float(np.percentile(valid_val_scores, config.detection.threshold_percentile))
    flags = np.isfinite(test_scores) & (test_scores > threshold) & (counts >= config.detection.min_score_count)

    frame = pd.DataFrame(
        {
            "timestep": np.arange(len(test_scores)),
            "score": test_scores,
            "window_count": counts,
            "is_anomaly": flags.astype(int),
        }
    )
    if bundle.test_labels is not None:
        frame["label"] = bundle.test_labels.astype(int)

    scores_path = output_dir / "anomaly_scores.csv"
    frame.to_csv(scores_path, index=False)

    summary = _build_summary(frame, threshold)
    with (output_dir / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if make_plots:
        plot_dir = ensure_dir(output_dir / "plots")
        plot_time_series_scores(
            values=bundle.test_raw,
            scores=test_scores,
            threshold=threshold,
            labels=bundle.test_labels,
            output_path=plot_dir / "test_scores.png",
            feature_name=bundle.feature_names[0],
        )
        plot_score_distribution(test_scores, threshold, plot_dir / "score_distribution.png")

    return scores_path


def _score_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset,
    series_length: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    window_errors = []
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="score"):
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            if getattr(model, "task", "forecast") == "reconstruct":
                target = x
                offset = 0
                window_length = sequence_length
            else:
                target = y
                offset = sequence_length
                window_length = y.size(1)
            errors = (prediction - target).pow(2).mean(dim=2).detach().cpu().numpy()
            window_errors.append(errors)

    errors = np.concatenate(window_errors, axis=0)
    return aggregate_window_errors(
        errors,
        starts=dataset.starts,
        series_length=series_length,
        window_length=window_length,
        offset=offset,
    )


def _build_summary(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    summary: dict[str, object] = {
        "threshold": threshold,
        "rows_scored": int(frame["score"].notna().sum()),
        "anomaly_points": int(frame["is_anomaly"].sum()),
    }
    if "label" not in frame:
        return summary

    valid = frame["score"].notna()
    y_true = frame.loc[valid, "label"].astype(bool)
    y_pred = frame.loc[valid, "is_anomaly"].astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    summary.update(
        {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score telemetry and flag anomalies.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--make-plots", action="store_true", help="Save diagnostic plots next to the checkpoint outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_path = run_inference(args.checkpoint, make_plots=args.make_plots)
    print(f"Saved anomaly scores to {scores_path}")


if __name__ == "__main__":
    main()

