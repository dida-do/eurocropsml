"""Main EuroCrops dataset class."""

import logging
from functools import partial
from pathlib import Path
from typing import Literal, cast

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
from eurocropsml.dataset.utils import (
    MMapStore,
    _pad_missing_dates,
    _unique_dates,
    pad_seq_to_366,
)

logger = logging.getLogger(__name__)


NORMALIZING_FACTOR_S1 = 1 / 51
NORMALIZING_FACTOR_S2 = 1e-4
EPSILON = 1e-6


class EuroCropsDataset(Dataset[LabelledData]):
    """PyTorch dataset class for the EuroCropsML dataset.

    Args:
        file_dict: Dictionary of file paths (one file per parcel/time series) for each satellite.
        encode: Encoding used to encode the classes into integers.
        mmap_store: Instance of memory map store.
        config: EuroCropsDatasetConfig instance.
        preprocess_config: EuroCropsDatasetPreprocessConfig instance.
        pad_seq_to_366: If sequence should be padded to 366 days.
        padding_value: Padding value used for padding data sequences.
    """

    def __init__(
        self,
        file_dict: dict[str, list[Path]],
        encode: dict[int, int],
        mmap_store: MMapStore,
        config: EuroCropsDatasetConfig,
        preprocess_config: EuroCropsDatasetPreprocessConfig,
        pad_seq_to_366: bool = False,
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.file_dict = file_dict
        self.mmap_store = mmap_store

        self.encode = encode
        self.config = config
        self.preprocess_config = preprocess_config
        self.pad_seq_to_366 = pad_seq_to_366
        self.padding_value = padding_value

        if "S2" in self.config.data_sources:
            band_names = cast(list[str], self.config.s2_bands)

            if self.config.remove_s2_bands is not None:
                self.keep_band_idxs: list[int] | None = []
                self.s2_data_bands: list[str] | None = []
                for band_idx, band in enumerate(band_names):
                    if band not in self.config.remove_s2_bands:
                        self.keep_band_idxs.append(band_idx)
                        self.s2_data_bands.append(band)
            else:
                self.keep_band_idxs = None
                self.s2_data_bands = band_names
        else:
            self.keep_band_idxs = None
            self.s2_data_bands = None
        if "S1" in self.config.data_sources:
            self.s1_data_bands: list[str] | None = self.config.s1_bands
        else:
            self.s1_data_bands = None

    @staticmethod
    def _format_dates(
        preprocess_config: EuroCropsDatasetPreprocessConfig,
        s2_data_bands: list[str] | None,
        date_type: Literal["day", "month"],
        arrays_dict: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:

        match date_type:
            case "day":
                # "-1" because days of the year should start at 0 for pos_encoding
                for satellite, dates in arrays_dict["dates"].items():
                    arrays_dict["dates"][satellite] = (
                        pd.to_datetime(dates).day_of_year.to_numpy() - 1
                    )
            case "month":
                for satellite, data in arrays_dict["data"].items():
                    dates = arrays_dict["dates"][satellite]
                    dates_month = pd.to_datetime(dates).month.to_numpy() - 1
                    unique, unique_indices, unique_counts = np.unique(
                        dates_month, return_index=True, return_counts=True
                    )
                    month_data = np.zeros((12, *data.shape[1:]))
                    for month, count, idx in zip(unique, unique_counts, unique_indices):
                        if count == 1:
                            month_data[month] = data[idx]
                        elif satellite == "S2":
                            s2_data_bands = cast(list[str], s2_data_bands)
                            try:
                                cloud_probs = np.apply_along_axis(
                                    partial(
                                        find_clouds,
                                        band4_idx=s2_data_bands.index("04"),
                                        preprocess_config=preprocess_config,
                                    ),
                                    1,
                                    data[idx : idx + count],
                                )
                            except ValueError as err:
                                raise ValueError(
                                    "Band 4 cannot be excluded if date_type is 'months'"
                                ) from err
                            month_data[month] = data[idx + np.argmin(cloud_probs)]
                    arrays_dict["data"][satellite] = month_data
                    arrays_dict["dates"][satellite] = np.arange(12)
        return arrays_dict

    def __getitem__(
        self,
        idx: int,
    ) -> LabelledData:
        f = {key: paths[idx] for key, paths in self.file_dict.items()}

        arrays_dict = self.mmap_store[f]

        arrays_dict = self._format_dates(
            self.preprocess_config,
            self.s2_data_bands,
            self.config.date_type,
            arrays_dict,
        )

        np_data_dict = arrays_dict.pop("data")
        if self.keep_band_idxs is not None:
            np_data_dict["S2"] = np_data_dict["S2"][:, self.keep_band_idxs]

        meta_data: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            array_name: {key: torch.tensor(np_array) for key, np_array in meta_array.items()}
            for array_name, meta_array in arrays_dict.items()
        }
        # center is always the same, replace by first value
        meta_data["center"] = next(iter(meta_data["center"].values()))
        # swap to lat, lon
        center_tensor = cast(torch.Tensor, meta_data["center"])
        meta_data["center"] = torch.flip(center_tensor, [0])

        # normalization and scaling to (0,1]
        if self.config.normalize:
            if "S1" in np_data_dict:
                # range before normalization and scaling: [-50.0,1.0]
                normalized_s1 = (np_data_dict["S1"] + 50) * NORMALIZING_FACTOR_S1
                np_data_dict["S1"] = normalized_s1 * (1 - EPSILON) + EPSILON
            if "S2" in np_data_dict:
                # range before normalization and scaling: [0.0, 1.0e+4]
                normalized_s2 = np_data_dict["S2"] * NORMALIZING_FACTOR_S2
                np_data_dict["S2"] = normalized_s2 * (1 - EPSILON) + EPSILON

        if self.pad_seq_to_366:
            for satellite, value_array in np_data_dict.items():
                np_data_dict[satellite] = pad_seq_to_366(
                    value_array,
                    meta_data["dates"][satellite],  # type: ignore[index]
                    self.padding_value,
                )
            meta_data["dates"] = _unique_dates(
                cast(dict[str, torch.Tensor], meta_data["dates"]), f.keys()
            )
            np_data: np.ndarray = np.hstack(list(np_data_dict.values()))

        elif len(f) == 2:  # if both S1 and S2 are used and no padding to 366 day
            all_dates: torch.Tensor = _unique_dates(
                cast(dict[str, torch.Tensor], meta_data["dates"]), f.keys()
            )
            np_data = _pad_missing_dates(
                np_data_dict,
                cast(dict[str, torch.Tensor], meta_data["dates"]),
                all_dates,
                len(self.s1_data_bands),  # type: ignore[arg-type]
                len(self.s2_data_bands),  # type: ignore[arg-type]
                self.padding_value,
            )
            meta_data["dates"] = all_dates  # only keep full range of dates
        else:
            meta_data["dates"] = _unique_dates(
                cast(dict[str, torch.Tensor], meta_data["dates"]), f.keys()
            )
            np_data = np.array(list(np_data_dict.values()))
            np_data = np.squeeze(np_data, axis=0)  # squeeze sensor dimension

        tensor_data = torch.tensor(np_data, dtype=torch.float)

        # get class from any filepath
        filepath: Path = next(iter(f.values()))
        y = int(filepath.stem.split("_")[-1])

        # encode class
        y = self.encode[y]
        target = torch.tensor(y)

        return LabelledData(
            DataItem(data=tensor_data, meta_data=cast(dict[str, torch.Tensor], meta_data)), target
        )

    def __len__(self) -> int:
        return max(len(files) for files in self.file_dict.values())
