import torch

from deep_space_navigation.config import ModelConfig
from deep_space_navigation.models import build_model


def test_patch_tst_forecaster_shape():
    config = ModelConfig(type="patch_tst", d_model=16, n_heads=4, num_layers=1, patch_length=4, patch_stride=2)
    model = build_model(config, n_features=3, sequence_length=12, prediction_horizon=5)
    output = model(torch.randn(2, 12, 3))

    assert tuple(output.shape) == (2, 5, 3)


def test_lstm_autoencoder_shape():
    config = ModelConfig(type="lstm_autoencoder", lstm_hidden_size=12, lstm_latent_size=6)
    model = build_model(config, n_features=3, sequence_length=12, prediction_horizon=5)
    output = model(torch.randn(2, 12, 3))

    assert tuple(output.shape) == (2, 12, 3)

