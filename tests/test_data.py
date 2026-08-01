import numpy as np

from deep_space_navigation.data import ForecastWindowDataset, aggregate_window_errors


def test_forecast_window_dataset_shapes():
    values = np.arange(60, dtype=np.float32).reshape(20, 3)
    dataset = ForecastWindowDataset(values, sequence_length=5, prediction_horizon=2, stride=1)

    x, y = dataset[0]

    assert len(dataset) == 14
    assert tuple(x.shape) == (5, 3)
    assert tuple(y.shape) == (2, 3)


def test_aggregate_window_errors_averages_overlapping_targets():
    errors = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    scores, counts = aggregate_window_errors(errors, starts=[0, 1], series_length=5, window_length=2, offset=1)

    assert np.isnan(scores[0])
    assert scores[1] == 1.0
    assert scores[2] == 4.0
    assert scores[3] == 7.0
    assert counts.tolist() == [0, 1, 2, 1, 0]

