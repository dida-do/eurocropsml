from pathlib import Path

import numpy as np
import pytest

from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.dataset.preprocess import _filter_clouds, _filter_padding


@pytest.fixture
def config() -> EuroCropsDatasetPreprocessConfig:
    return EuroCropsDatasetPreprocessConfig(
        raw_data_dir=Path("raw_data_dir"), preprocess_dir=Path("preprocess_dir")
    )


@pytest.fixture
def test_arrays() -> tuple[np.ndarray, np.ndarray]:
    test_data = np.stack(
        [
            np.array([0] * 4),
            np.array([0, 0, 0, 2600]),
            np.array([0, 0, 0, 2400]),
            np.array([0, 0, 0, 600]),
        ]
    )
    test_dates = np.arange(0, len(test_data))
    return test_data, test_dates


def test_filter_padding(
    test_arrays: tuple[np.ndarray, np.ndarray], config: EuroCropsDatasetPreprocessConfig
) -> None:
    test_data, test_dates = test_arrays

    filtered_test_data, filtered_test_dates = _filter_padding(test_data, test_dates)

    assert filtered_test_data.shape == (test_data.shape[0] - 1, *test_data.shape[1:])
    assert filtered_test_dates.shape == (test_dates.shape[0] - 1, *test_dates.shape[1:])
    assert np.equal(np.setdiff1d(test_dates, filtered_test_dates), np.array([0]))

    filtered_test_data, filtered_test_dates = _filter_clouds(
        filtered_test_data, filtered_test_dates, config
    )

    assert filtered_test_data.shape == (test_data.shape[0] - 3, *test_data.shape[1:])
    assert filtered_test_dates.shape == (test_dates.shape[0] - 3, *test_dates.shape[1:])
    assert np.array_equal(np.setdiff1d(test_dates, filtered_test_dates), np.array([0, 1, 2]))
