from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurocropsml.dataset.config import S2_BANDS, EuroCropsDatasetPreprocessConfig
from eurocropsml.dataset.dataset import EuroCropsDataset


@pytest.fixture
def preprocess_config() -> EuroCropsDatasetPreprocessConfig:
    return EuroCropsDatasetPreprocessConfig(
        raw_data_dir=Path("raw_data_dir"), preprocess_dir=Path("preprocess_dir")
    )


@pytest.fixture
def leap_test_arrays_dict() -> dict[str, dict[str, np.ndarray]]:
    return {
        "data": {"S1": np.ones((366, 2)), "S2": np.ones((366, 13))},
        "dates": {
            "S1": pd.date_range(start="2020-01-01", end="2020-12-31").to_numpy(),
            "S2": pd.date_range(start="2020-01-01", end="2020-12-31").to_numpy(),
        },
    }


@pytest.fixture
def test_arrays_dict() -> dict[str, dict[str, np.ndarray]]:
    return {
        "data": {"S1": np.ones((366, 2)), "S2": np.ones((366, 13))},
        "dates": {
            "S1": pd.date_range(start="2021-01-01", end="2021-12-31").to_numpy(),
            "S2": pd.date_range(start="2021-01-01", end="2021-12-31").to_numpy(),
        },
    }


def test_format_dates_day(
    test_arrays_dict: dict[str, dict[str, np.ndarray]],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> None:
    arrays_dict = EuroCropsDataset._format_dates(
        date_type="day",
        arrays_dict=test_arrays_dict,
        preprocess_config=preprocess_config,
        s2_data_bands=S2_BANDS,
    )
    for satellite in arrays_dict["data"]:
        data = arrays_dict["data"][satellite]
        dates = arrays_dict["dates"][satellite]
        assert data.shape == test_arrays_dict["data"][satellite].shape
        assert np.max(dates) == 364
        assert np.min(dates) == 0
        assert np.array_equal(np.unique(dates), dates)


def test_format_dates_day_leapyear(
    leap_test_arrays_dict: dict[str, dict[str, np.ndarray]],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> None:
    arrays_dict = EuroCropsDataset._format_dates(
        date_type="day",
        arrays_dict=leap_test_arrays_dict,
        preprocess_config=preprocess_config,
        s2_data_bands=S2_BANDS,
    )
    for satellite in arrays_dict["data"]:
        data = arrays_dict["data"][satellite]
        dates = arrays_dict["dates"][satellite]
        assert data.shape == leap_test_arrays_dict["data"][satellite].shape
        assert np.max(dates) == 365
        assert np.min(dates) == 0
        assert np.array_equal(np.unique(dates), dates)


def test_format_dates_month(
    test_arrays_dict: dict[str, dict[str, np.ndarray]],
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> None:
    arrays_dict = EuroCropsDataset._format_dates(
        date_type="month",
        arrays_dict=test_arrays_dict,
        preprocess_config=preprocess_config,
        s2_data_bands=S2_BANDS,
    )
    for satellite in arrays_dict["data"]:
        data = arrays_dict["data"][satellite]
        dates = arrays_dict["dates"][satellite]
        assert data.shape == (12, test_arrays_dict["data"][satellite].shape[1])
        assert np.max(dates) == 11
        assert np.min(dates) == 0
