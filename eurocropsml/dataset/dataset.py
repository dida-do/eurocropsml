"""Main EuroCrops dataset class."""

import logging
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eurocropsml.dataset.base import DataItem, LabelledData
from eurocropsml.dataset.config import (
    EuroCropsDatasetConfig,
    EuroCropsDatasetPreprocessConfig,
)
from eurocropsml.dataset.preprocess import find_clouds
from eurocropsml.dataset.utils import MMapStore, pad_seq_to_366

logger = logging.getLogger(__name__)


NORMALIZING_FACTOR = 1e-4

EUROCROPS_S2BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]


class EuroCropsDataset(Dataset[LabelledData]):
    """PyTorch dataset class for the EuroCropsML dataset.

    Args:
        file_list: List of file paths (one file per parcel/time series).
        encode: Encoding used to encode the classes into integers.
        mmap_store: Instance of memory map store.
        config: EuroCropsDatasetConfig instance.
        pad_seq_to_366: If sequence should be padded to 366 days
            This is only used for TIML with an encoder.
    """

    def __init__(
        self,
        file_list: list[Path],
        encode: dict[int, int],
        mmap_store: MMapStore,
        config: EuroCropsDatasetConfig,
        preprocess_config: EuroCropsDatasetPreprocessConfig,
        pad_seq_to_366: bool = False,
    ) -> None:
        super().__init__()

        self.file_list = file_list
        self.mmap_store = mmap_store

        self.encode = encode
        self.config = config
        self.preprocess_config = preprocess_config
        self.pad_seq_to_366 = pad_seq_to_366

        if self.config.remove_bands is not None:
            self.keep_band_idxs: list[int] | None = []
            self.data_bands: list[str] = []
            for band_idx, band in enumerate(EUROCROPS_S2BANDS):
                if band not in self.config.remove_bands:
                    self.keep_band_idxs.append(band_idx)
                    self.data_bands.append(band)
        else:
            self.keep_band_idxs = None
            self.data_bands = EUROCROPS_S2BANDS

    @staticmethod
    def _format_dates(
        preprocess_config: EuroCropsDatasetPreprocessConfig,
        date_type: Literal["day", "month"],
        arrays_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        data, dates = arrays_dict["data"], arrays_dict["dates"]

        match date_type:
            case "day":
                # "-1" because days of the year should start at 0 for pos_encoding
                arrays_dict["dates"] = pd.to_datetime(dates).day_of_year.to_numpy() - 1
            case "month":
                dates_month = pd.to_datetime(dates).month.to_numpy() - 1
                unique, unique_indices, unique_counts = np.unique(
                    dates_month, return_index=True, return_counts=True
                )
                month_data = np.zeros((12, *data.shape[1:]))
                for month, count, idx in zip(unique, unique_counts, unique_indices):
                    if count == 1:
                        month_data[month] = data[idx]
                    else:
                        try:
                            cloud_probs = np.apply_along_axis(
                                partial(
                                    find_clouds,
                                    band4_idx=EUROCROPS_S2BANDS.index("B4"),
                                    preprocess_config=preprocess_config,
                                ),
                                1,
                                data[idx : idx + count],
                            )
                        except ValueError:
                            raise ValueError("Band 4 cannot be excluded if date_type is 'months'")
                        month_data[month] = data[idx + np.argmin(cloud_probs)]
                arrays_dict["data"], arrays_dict["dates"] = month_data, np.arange(12)
        return arrays_dict

    def __getitem__(
        self,
        idx: int,
    ) -> LabelledData:
        f = self.file_list[idx]
        arrays_dict = self.mmap_store[f]

        arrays_dict = self._format_dates(self.preprocess_config, self.config.date_type, arrays_dict)

        np_data = arrays_dict.pop("data")
        if self.keep_band_idxs is not None:
            np_data = np_data[:, self.keep_band_idxs]

        meta_data: dict[str, torch.Tensor] = {
            array_name: torch.tensor(np_array) for array_name, np_array in arrays_dict.items()
        }
        if self.pad_seq_to_366:
            np_data = pad_seq_to_366(np_data, meta_data["dates"])

        tensor_data = torch.tensor(np_data, dtype=torch.float)
        if self.config.normalize:
            tensor_data = torch.mul(tensor_data, NORMALIZING_FACTOR)

        y = int(Path(f).stem.split("_")[-1])
        # encode class
        y = self.encode[y]
        target = torch.tensor(y)

        return LabelledData(DataItem(data=tensor_data, meta_data=meta_data), target)

    def __len__(self) -> int:
        return len(self.file_list)
