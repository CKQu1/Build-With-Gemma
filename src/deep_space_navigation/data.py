from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainingConfig


@dataclass
class SeriesScaler:
    method: str
    center: np.ndarray | None = None
    scale: np.ndarray | None = None

    @classmethod
    def fit(cls, values: np.ndarray, method: str) -> "SeriesScaler":
        method = method.lower()
        if method == "none":
            return cls(method=method)

        if method == "standard":
            center = values.mean(axis=0)
            scale = values.std(axis=0)
        elif method == "minmax":
            center = values.min(axis=0)
            scale = values.max(axis=0) - center
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

        scale = np.where(scale == 0, 1.0, scale)
        return cls(method=method, center=center.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return values.astype(np.float32)
        if self.center is None or self.scale is None:
            raise ValueError("Scaler has not been fitted.")
        return ((values - self.center) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return values.astype(np.float32)
        if self.center is None or self.scale is None:
            raise ValueError("Scaler has not been fitted.")
        return (values * self.scale + self.center).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "center": None if self.center is None else self.center.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "SeriesScaler":
        center = state.get("center")
        scale = state.get("scale")
        return cls(
            method=str(state["method"]),
            center=None if center is None else np.asarray(center, dtype=np.float32),
            scale=None if scale is None else np.asarray(scale, dtype=np.float32),
        )


@dataclass
class DataBundle:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_raw: np.ndarray
    val_raw: np.ndarray
    test_raw: np.ndarray
    test_labels: np.ndarray | None
    scaler: SeriesScaler
    feature_names: list[str]


class ForecastWindowDataset(Dataset):
    """Sliding windows for history-to-future sequence forecasting."""

    def __init__(self, values: np.ndarray, sequence_length: int, prediction_horizon: int, stride: int = 1):
        if values.ndim != 2:
            raise ValueError("values must have shape (time, features)")
        if sequence_length <= 0 or prediction_horizon <= 0:
            raise ValueError("sequence_length and prediction_horizon must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.values = values.astype(np.float32)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        last_start = len(values) - sequence_length - prediction_horizon
        self.starts = np.arange(0, max(last_start + 1, 0), stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        input_end = start + self.sequence_length
        target_end = input_end + self.prediction_horizon
        x = self.values[start:input_end]
        y = self.values[input_end:target_end]
        return torch.from_numpy(x), torch.from_numpy(y)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(config: DataConfig, scaler_state: dict[str, object] | None = None) -> DataBundle:
    if config.source == "synthetic":
        full, labels, feature_names = _make_synthetic_series(config)
        train_raw, val_raw, test_raw = _temporal_split(full, config)
        _, _, test_labels = _temporal_split(labels[:, None].astype(np.float32), config)
        test_labels = test_labels[:, 0].astype(bool)
    elif config.source == "csv":
        full, feature_names = _load_csv_series(config)
        train_raw, val_raw, test_raw = _temporal_split(full, config)
        test_labels = None
    elif config.source == "nasa":
        train_raw, val_raw, test_raw, test_labels, feature_names = _load_nasa_series(config)
    else:
        raise ValueError(f"Unsupported data source: {config.source}")

    train_raw = _impute_array(train_raw, config.imputation)
    val_raw = _impute_array(val_raw, config.imputation)
    test_raw = _impute_array(test_raw, config.imputation)

    scaler = SeriesScaler.from_state_dict(scaler_state) if scaler_state else SeriesScaler.fit(train_raw, config.scaling)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)

    return DataBundle(
        train=train,
        val=val,
        test=test,
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        test_labels=test_labels,
        scaler=scaler,
        feature_names=feature_names,
    )


def create_datasets(config: DataConfig, bundle: DataBundle) -> dict[str, ForecastWindowDataset]:
    datasets = {
        "train": ForecastWindowDataset(bundle.train, config.sequence_length, config.prediction_horizon, config.stride),
        "val": ForecastWindowDataset(bundle.val, config.sequence_length, config.prediction_horizon, config.stride),
        "test": ForecastWindowDataset(bundle.test, config.sequence_length, config.prediction_horizon, config.stride),
    }
    empty = [name for name, dataset in datasets.items() if len(dataset) == 0]
    if empty:
        min_required = config.sequence_length + config.prediction_horizon
        raise ValueError(f"Not enough timesteps for windows in {empty}. Need at least {min_required} rows per split.")
    return datasets


def create_dataloaders(
    datasets: dict[str, ForecastWindowDataset],
    training: TrainingConfig,
) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=training.batch_size,
            shuffle=True,
            num_workers=training.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=training.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


def aggregate_window_errors(
    window_errors: np.ndarray,
    starts: Iterable[int],
    series_length: int,
    window_length: int,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    score_sum = np.zeros(series_length, dtype=np.float64)
    score_count = np.zeros(series_length, dtype=np.int64)

    for row, start in zip(window_errors, starts):
        target_start = int(start) + offset
        target_end = min(target_start + window_length, series_length)
        usable = max(target_end - target_start, 0)
        if usable == 0:
            continue
        score_sum[target_start:target_end] += row[:usable]
        score_count[target_start:target_end] += 1

    scores = np.full(series_length, np.nan, dtype=np.float32)
    valid = score_count > 0
    scores[valid] = (score_sum[valid] / score_count[valid]).astype(np.float32)
    return scores, score_count


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {values.shape}")
    return values.astype(np.float32)


def _impute_array(values: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    frame = pd.DataFrame(values)
    if method == "none":
        return values.astype(np.float32)
    if method == "interpolate":
        frame = frame.interpolate(limit_direction="both").ffill().bfill()
    elif method == "mean":
        frame = frame.fillna(frame.mean(numeric_only=True))
    elif method == "median":
        frame = frame.fillna(frame.median(numeric_only=True))
    else:
        raise ValueError(f"Unsupported imputation method: {method}")
    return frame.to_numpy(dtype=np.float32)


def _temporal_split(values: np.ndarray, config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = config.train_fraction + config.val_fraction + config.test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")
    n_rows = len(values)
    train_end = int(n_rows * config.train_fraction)
    val_end = train_end + int(n_rows * config.val_fraction)
    return values[:train_end], values[train_end:val_end], values[val_end:]


def _load_csv_series(config: DataConfig) -> tuple[np.ndarray, list[str]]:
    if not config.csv_path:
        raise ValueError("csv_path is required when data.source is 'csv'")
    frame = pd.read_csv(config.csv_path)
    if config.timestamp_col and config.timestamp_col in frame:
        frame = frame.sort_values(config.timestamp_col)

    if config.feature_cols:
        feature_cols = config.feature_cols
    else:
        feature_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
        if config.timestamp_col in feature_cols:
            feature_cols.remove(config.timestamp_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in the CSV file.")
    return frame[feature_cols].to_numpy(dtype=np.float32), feature_cols


def _load_nasa_series(config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    base = Path(config.data_dir)
    train_path = _find_channel_file(base, "train", config.channel_id)
    test_path = _find_channel_file(base, "test", config.channel_id)
    train_full = _as_2d(np.load(train_path))
    test_full = _as_2d(np.load(test_path))

    if config.use_official_nasa_test:
        train_raw, val_raw = _split_train_val(train_full, config.train_fraction, config.val_fraction)
        test_raw = test_full
        test_labels = _load_nasa_labels(base, config.channel_id, len(test_full))
    else:
        combined = np.concatenate([train_full, test_full], axis=0)
        train_raw, val_raw, test_raw = _temporal_split(combined, config)
        test_labels = None

    feature_names = [f"sensor_{idx}" for idx in range(train_full.shape[1])]
    return train_raw, val_raw, test_raw, test_labels, feature_names


def _split_train_val(values: np.ndarray, train_fraction: float, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    denominator = train_fraction + val_fraction
    if denominator <= 0:
        raise ValueError("train_fraction + val_fraction must be positive")
    train_end = int(len(values) * (train_fraction / denominator))
    return values[:train_end], values[train_end:]


def _find_channel_file(base: Path, split: str, channel_id: str) -> Path:
    candidates = [
        base / split / f"{channel_id}.npy",
        base / "data" / split / f"{channel_id}.npy",
        base / "SMAP" / split / f"{channel_id}.npy",
        base / "MSL" / split / f"{channel_id}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(base.rglob(f"{channel_id}.npy"))
    split_matches = [path for path in matches if split.lower() in {part.lower() for part in path.parts}]
    if split_matches:
        return split_matches[0]

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {split} file for channel {channel_id}. Tried:\n{searched}")


def _load_nasa_labels(base: Path, channel_id: str, length: int) -> np.ndarray | None:
    candidates = [base / "labeled_anomalies.csv", base / "data" / "labeled_anomalies.csv"]
    label_path = next((path for path in candidates if path.exists()), None)
    if label_path is None:
        matches = list(base.rglob("labeled_anomalies.csv"))
        label_path = matches[0] if matches else None
    if label_path is None:
        return None

    labels = np.zeros(length, dtype=bool)
    frame = pd.read_csv(label_path)
    if "chan_id" not in frame or "anomaly_sequences" not in frame:
        return labels

    rows = frame[frame["chan_id"].astype(str) == str(channel_id)]
    for sequence_text in rows["anomaly_sequences"].dropna():
        for start, end in _parse_anomaly_sequences(sequence_text):
            start = max(0, int(start))
            end = min(length - 1, int(end))
            if end >= start:
                labels[start : end + 1] = True
    return labels


def _parse_anomaly_sequences(value: str) -> list[tuple[int, int]]:
    parsed = ast.literal_eval(value)
    return [(int(start), int(end)) for start, end in parsed]


def _make_synthetic_series(config: DataConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(12345)
    t = np.arange(config.synthetic_length, dtype=np.float32)
    features = []
    for idx in range(config.synthetic_features):
        seasonal = np.sin(t / (18 + idx * 4)) + 0.35 * np.cos(t / (9 + idx))
        trend = (idx + 1) * 0.0008 * t
        noise = rng.normal(0, 0.05 + idx * 0.005, size=config.synthetic_length)
        features.append(seasonal + trend + noise)
    values = np.stack(features, axis=1).astype(np.float32)

    labels = np.zeros(config.synthetic_length, dtype=bool)
    n_anomalies = max(1, int(config.synthetic_length * config.synthetic_anomaly_fraction))
    starts = rng.choice(np.arange(config.sequence_length, config.synthetic_length - 30), size=n_anomalies, replace=False)
    for start in starts:
        width = int(rng.integers(5, 24))
        end = min(start + width, config.synthetic_length)
        affected = int(rng.integers(0, config.synthetic_features))
        values[start:end, affected] += rng.normal(2.5, 0.4)
        labels[start:end] = True

    feature_names = [f"synthetic_sensor_{idx}" for idx in range(config.synthetic_features)]
    return values, labels, feature_names

