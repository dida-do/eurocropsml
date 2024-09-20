import numpy as np
import pytest
import torch

from eurocropsml.dataset.utils import pad_seq_to_366


@pytest.fixture
def test_arrays() -> tuple[np.ndarray, torch.Tensor]:
    test_data = np.stack(
        [
            np.array([0] * 4),
            np.array([0, 0, 3000, 100]),
            np.array([0, 0, 0, 2600]),
            np.array([0, 0, 0, 2400]),
            np.array([0, 0, 0, 600]),
        ]
    )
    test_dates = np.arange(0, len(test_data))
    return test_data, torch.from_numpy(test_dates)


def test_pad_seq_to_366(test_arrays: tuple[np.ndarray, torch.Tensor]) -> None:
    test_data, test_dates = test_arrays
    np_data = pad_seq_to_366(test_data, test_dates)

    assert np_data.shape == (366, test_data.shape[1])
