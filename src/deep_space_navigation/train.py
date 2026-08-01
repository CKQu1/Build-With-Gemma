from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config, save_config, to_dict
from .data import create_dataloaders, create_datasets, prepare_data, seed_everything
from .models import build_model
from .training_utils import EarlyStopping, build_loss, ensure_dir, select_device


def train(config: ExperimentConfig) -> Path:
    seed_everything(config.training.seed)
    output_dir = ensure_dir(Path(config.output.root_dir) / config.output.experiment_name)
    save_config(config, output_dir / "resolved_config.yaml")

    bundle = prepare_data(config.data)
    datasets = create_datasets(config.data, bundle)
    loaders = create_dataloaders(datasets, config.training)
    device = select_device(config.training.device)
    model = build_model(
        config.model,
        n_features=bundle.train.shape[1],
        sequence_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon,
    ).to(device)

    criterion = build_loss(config.training.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    stopper = EarlyStopping(config.training.early_stopping_patience)
    history: list[dict[str, float | int]] = []
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        val_loss = _run_epoch(model=model, loader=loaders["val"], criterion=criterion, device=device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        tqdm.write(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss <= stopper.best_loss:
            _save_checkpoint(best_path, model, config, bundle, epoch, val_loss)
        if stopper.step(val_loss):
            tqdm.write(f"early stopping after {epoch} epochs")
            break

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return best_path


def _target_for(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x if getattr(model, "task", "forecast") == "reconstruct" else y


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_rows = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False, desc="train" if is_training else "eval"):
            x = x.to(device)
            y = y.to(device)
            target = _target_for(model, x, y)
            prediction = model(x)
            loss = criterion(prediction, target)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm and gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            batch_size = x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_rows += batch_size

    return total_loss / max(total_rows, 1)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: ExperimentConfig,
    bundle,
    epoch: int,
    val_loss: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": to_dict(config),
        "scaler": bundle.scaler.state_dict(),
        "feature_names": bundle.feature_names,
        "n_features": bundle.train.shape[1],
        "epoch": epoch,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch telemetry anomaly detector.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_path = train(load_config(args.config))
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
