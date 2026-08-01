from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        input_length: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.task = "forecast"
        self.horizon = horizon
        self.n_features = n_features
        self.input_projection = nn.Linear(n_features, d_model)
        self.position = PositionalEncoding(d_model, max_len=input_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * n_features),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values)
        hidden = self.position(hidden)
        encoded = self.encoder(hidden)
        forecast = self.head(encoded[:, -1])
        return forecast.view(values.size(0), self.horizon, self.n_features)


class PatchTSTForecaster(nn.Module):
    """PatchTST-inspired channel-independent Transformer forecaster."""

    def __init__(
        self,
        n_features: int,
        input_length: int,
        horizon: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if patch_length > input_length:
            raise ValueError("patch_length cannot be greater than sequence_length")
        self.task = "forecast"
        self.n_features = n_features
        self.horizon = horizon
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.n_patches = ((input_length - patch_length) // patch_stride) + 1

        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, _, n_features = values.shape
        patches = values.transpose(1, 2).unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)
        patches = patches.contiguous().view(batch_size * n_features, self.n_patches, self.patch_length)
        hidden = self.patch_projection(patches) + self.position
        encoded = self.encoder(hidden)
        pooled = encoded.mean(dim=1)
        forecast = self.head(pooled)
        return forecast.view(batch_size, n_features, self.horizon).transpose(1, 2)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, input_length: int, hidden_size: int, latent_size: int, dropout: float):
        super().__init__()
        self.task = "reconstruct"
        self.input_length = input_length
        self.n_features = n_features
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, n_features)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(values)
        latent = torch.tanh(self.to_latent(hidden[-1]))
        repeated = self.from_latent(latent).unsqueeze(1).repeat(1, self.input_length, 1)
        decoded, _ = self.decoder(self.dropout(repeated))
        return self.output_projection(decoded)


def build_model(config: ModelConfig, n_features: int, sequence_length: int, prediction_horizon: int) -> nn.Module:
    model_type = config.type.lower()
    if model_type == "patch_tst":
        return PatchTSTForecaster(
            n_features=n_features,
            input_length=sequence_length,
            horizon=prediction_horizon,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if model_type == "transformer":
        return TransformerForecaster(
            n_features=n_features,
            input_length=sequence_length,
            horizon=prediction_horizon,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if model_type == "lstm_autoencoder":
        return LSTMAutoencoder(
            n_features=n_features,
            input_length=sequence_length,
            hidden_size=config.lstm_hidden_size,
            latent_size=config.lstm_latent_size,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model type: {config.type}")

