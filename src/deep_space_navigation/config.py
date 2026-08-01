from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"
    data_dir: str = "data/raw/nasa"
    channel_id: str = "P-1"
    csv_path: str | None = None
    timestamp_col: str | None = None
    feature_cols: list[str] | None = None
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    use_official_nasa_test: bool = True
    sequence_length: int = 100
    prediction_horizon: int = 20
    stride: int = 1
    scaling: str = "standard"
    imputation: str = "interpolate"
    synthetic_length: int = 2500
    synthetic_features: int = 6
    synthetic_anomaly_fraction: float = 0.02


@dataclass
class ModelConfig:
    type: str = "patch_tst"
    d_model: int = 96
    n_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    patch_length: int = 16
    patch_stride: int = 8
    lstm_hidden_size: int = 96
    lstm_latent_size: int = 48


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss: str = "huber"
    early_stopping_patience: int = 5
    num_workers: int = 0
    device: str = "auto"
    gradient_clip_norm: float = 1.0


@dataclass
class DetectionConfig:
    threshold_percentile: float = 99.0
    min_score_count: int = 1


@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    experiment_name: str = "patchtst_smap_msl"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    detection: DetectionConfig
    output: OutputConfig


def _coerce_dataclass(cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any] | None) -> ExperimentConfig:
    raw = raw or {}
    return ExperimentConfig(
        data=_coerce_dataclass(DataConfig, raw.get("data")),
        model=_coerce_dataclass(ModelConfig, raw.get("model")),
        training=_coerce_dataclass(TrainingConfig, raw.get("training")),
        detection=_coerce_dataclass(DetectionConfig, raw.get("detection")),
        output=_coerce_dataclass(OutputConfig, raw.get("output")),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return config_from_dict(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)

