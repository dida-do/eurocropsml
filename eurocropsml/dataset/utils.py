"""Utilities for the dataset creation."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property, partial
from itertools import chain
from operator import itemgetter
from pathlib import Path
from statistics import median
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from mmap_ninja.ragged import RaggedMmap
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from eurocropsml.settings import Settings

logger = logging.getLogger(__name__)


def read_files(data_dir: Path) -> set[str]:
    """Method to read all `.npy` files in a directory.

    Args:
        data_dir: Path that contains `.npy` files where labels and data are stored.

    Returns:
        List of all `.npz` files that lie within the directory.
    """
    return {file.name for file in data_dir.iterdir() if file.suffix == ".npz"}


def _split_dataset(
    data_dir: Path,
    satellite: list[Literal["S1", "S2"]],
    year: str,
    pretrain_classes: set[int],
    finetune_classes: set[int] | None = None,
    regions: set[str] | None = None,
) -> tuple[dict[int, list[str]], dict[int, list[str]] | None]:
    file_set: set = set()
    for s in satellite:
        file_set.update(read_files(data_dir.joinpath(s, year)))

    # pre-filter by regions for region and regionclass split
    if regions is not None:
        file_set = {file for file in file_set if file.startswith(tuple(regions))}

    pretrain_dataset, finetune_dataset = _filter_classes(
        file_set,
        pretrain_classes,
        finetune_classes,
    )

    return pretrain_dataset, finetune_dataset


def _filter_classes(
    file_set: set[str],
    pretrain_classes: set[int],
    finetune_classes: set[int] | None,
) -> tuple[dict[int, list[str]], dict[int, list[str]] | None]:
    pretrain_dict: dict[int, list[str]] = defaultdict(list)
    finetune_dict: dict[int, list[str]] = defaultdict(list)
    pretrain_dict = {
        identifier: [file for file in file_set if Path(file).stem.split("_")[-1] == str(identifier)]
        for identifier in pretrain_classes
    }

    if finetune_classes is not None:
        finetune_dict = {
            identifier: [
                file for file in file_set if Path(file).stem.split("_")[-1] == str(identifier)
            ]
            for identifier in finetune_classes
        }
        return pretrain_dict, finetune_dict
    else:
        return pretrain_dict, None


def _order_classes(file_list: list[str]) -> dict:
    """Order file paths by classes.

    Args:
        file_list: List of file paths.

    Returns:
        Dictionary of file paths with classes as keys.
    """
    class_dict: dict[int, list[str]] = defaultdict(list)

    for file in file_list:
        c = int(Path(file).stem.split("_")[2])
        class_dict[c].append(str(file))

    return class_dict


ClassFilesDictT = dict[int, list[str]]
RegionFilesDictT = dict[str, list[str]]


def _create_final_dict(
    file_list: list[str],
    regions: set[str],
) -> dict[str, ClassFilesDictT]:
    file_dict: RegionFilesDictT = defaultdict(list)

    # create dictionary with regions as keys
    for key in regions:
        file_dict[key] = [file_name for file_name in file_list if file_name.startswith(key)]

    # remove empty keys
    file_dict = {region: files for region, files in file_dict.items() if files != []}

    # build nested dictionary
    # regions as keys in the outer dict
    # classes as keys in the inner dict
    return _create_nested_dict(file_dict)


RegionClassFilesDictT = dict[str, ClassFilesDictT]


def _create_nested_dict(data_dict: RegionFilesDictT) -> RegionClassFilesDictT:
    full_dict: RegionClassFilesDictT = defaultdict(dict)
    for key in data_dict:
        class_dict: ClassFilesDictT = defaultdict(list)
        for file in data_dict[key]:
            c = int(Path(file).stem.split("_")[2])
            class_dict[c].append(file)
        full_dict[key] = class_dict
    return full_dict


def _save_to_json(json_file: Path, data: dict) -> None:
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(json_file), "w") as outfile:
        json.dump(data, outfile)


def _save_to_dict(train: list[str], val: list[str], test: list[str] | None = None) -> dict:
    dict_data: dict[str, list[str]] = defaultdict(list)
    dict_data["train"] = train
    dict_data["val"] = val

    if test is not None:
        dict_data["test"] = test

    return dict_data


def _filter_regions(dataset: dict, regions: set) -> dict:
    for key, file_list in dataset.items():
        dataset[key] = [file for file in file_list if file.startswith(tuple(regions))]
    # remove empty keys
    dataset = {c: files for c, files in dataset.items() if files != []}

    return dataset


def _sample_max_samples(file_list: list[str], max_samples: int, seed: int) -> list[str]:
    class_dict = _order_classes(file_list)
    dataset_list = _downsample_class(class_dict, seed, n_samples=max_samples)

    return dataset_list


def _downsample_class(
    dataset_dict: dict[int, list[str]],
    seed: int,
    class_key: int | None = None,
    n_samples: int | None = None,
) -> list[str]:
    """Downsample class to n_samples.

    Args:
        dataset_dict: Dictionary with classes as keys and lists of file paths as values
        seed: Randoms seed
        class_key: Class to downsample. If None, all classes will be downsamples to n_samples.
        n_samples: Number to downsample the class to.
            If not specified, median of remaining classes will be used.

    Returns:
        List of file paths.

    Raises:
        ValueError: If neither class_key nor n_samples is speficied.

    """

    if class_key is None:
        if n_samples is None:
            raise ValueError("Please specify the number of n_samples or the class_key.")
        else:
            for key, files in dataset_dict.items():
                if len(files) > n_samples:
                    dataset_dict[key] = resample(
                        files,
                        replace=False,
                        n_samples=n_samples,
                        random_state=seed,
                    )

    elif class_key in dataset_dict:
        if n_samples is None:
            n_samples = int(
                median([len(val) for key, val in dataset_dict.items() if key != class_key])
            )
        if len(dataset_dict[class_key]) > n_samples:
            dataset_dict[class_key] = resample(
                dataset_dict[class_key],
                replace=False,
                n_samples=n_samples,
                random_state=seed,
            )

    return list(chain.from_iterable(dataset_dict.values()))


def _save_counts_to_csv(data_list: list[str], data_dir: Path, split: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dict = _order_classes(data_list)
    data_counts = {key: len(val) for key, val in data_dict.items()}
    data_df = pd.DataFrame(list(data_counts.items()))
    file_path = Path(data_dir / f"{split}_split.csv")
    data_df.to_csv(file_path, header=False)


def _create_finetune_set(
    finetune_dataset: dict[int, list[str]],
    split_path: Path,
    split: str,
    pretrain_list: list[str],
    num_samples: dict[str, str | int | list[int | str]],
    test_size: float,
    seed: int,
) -> None:
    finetune_list: list[str] = [
        value for values_list in finetune_dataset.values() for value in values_list
    ]

    finetune_list.sort()
    if set(pretrain_list) & set(finetune_list):
        raise Exception(
            f"There are {len((set(pretrain_list) & set(finetune_list)))} "
            "equal samples within upstream and downstream task."
        )

    finetune_train, finetune_val = train_test_split(
        finetune_list, test_size=2 * test_size, random_state=seed
    )
    new_test_size = np.around(test_size / (2 * test_size), 2)
    finetune_val, finetune_test = train_test_split(
        finetune_val, test_size=new_test_size, random_state=seed
    )
    if num_samples["validation"] != "all":
        num_samples["validation"] = int(cast(int, num_samples["validation"]))
    if num_samples["test"] != "all":
        num_samples["test"] = int(cast(int, num_samples["test"]))
    if isinstance(num_samples["validation"], int) and len(finetune_val) > num_samples["validation"]:
        finetune_val = resample(
            finetune_val,
            replace=False,
            n_samples=num_samples["validation"],
            random_state=seed,
        )
    if isinstance(num_samples["test"], int) and len(finetune_test) > num_samples["test"]:
        finetune_test = resample(
            finetune_test,
            replace=False,
            n_samples=num_samples["test"],
            random_state=seed,
        )

    sample_list: list[str | int]
    if isinstance(num_samples["train"], list):
        sample_list = num_samples["train"]
    else:
        sample_list = [cast(int, num_samples["train"])]

    if "all" in sample_list:
        _save_to_json(
            split_path.joinpath(f"{split}_split_all.json"),
            _save_to_dict(finetune_train, finetune_val, finetune_test),
        )
        sample_list.remove("all")
    for max_samples in sample_list:
        train = _sample_max_samples(finetune_train, max_samples, seed)  # type: ignore[arg-type]
        _save_to_json(
            split_path.joinpath(f"{split}_split_{max_samples}.json"),
            _save_to_dict(train, finetune_val, finetune_test),
        )

    _save_counts_to_csv(finetune_list, split_path.parents[0].joinpath("counts", "finetune"), split)


def _pad_missing_dates(
    data_dict: dict[str, np.ndarray],
    dates: dict[str, torch.Tensor],
    all_dates: torch.Tensor,
    s1_bands_len: int = 2,
    s2_bands_len: int = 13,
    padding_value: float = 0.0,
) -> np.ndarray:
    # Combined output array for S1 and S2 bands with padding values
    combined_data: np.ndarray = np.full(
        (len(all_dates), s1_bands_len + s2_bands_len), padding_value
    )

    # Find insertion indices for S1 and S2 bands
    s1_indices = np.searchsorted(all_dates, dates["S1"])
    s2_indices = np.searchsorted(all_dates, dates["S2"])

    # Insert data directly into the combined array for S1 and S2
    combined_data[s1_indices, :s1_bands_len] = data_dict["S1"]
    combined_data[s2_indices, s1_bands_len:] = data_dict["S2"]

    return combined_data


def pad_seq_to_366(seq: np.ndarray, dates: torch.Tensor, padding_value: float = -1.0) -> np.ndarray:
    """Pad sequence to 366 days.

    Args:
        seq: Tensor containing sequence data to be padded.
        dates: Tensor of matching size specifying the dates
            associated to each the sequences data point.
        padding_value: Value used for padding.

    Returns:
        A padded sequence data array with all missing dates filled in by a
            `padding_value`-mask value.

    """

    padded_seq = np.full((366, seq.shape[1]), padding_value, dtype=seq.dtype)

    date_indices = dates.to(torch.int32).numpy()

    padded_seq[date_indices] = seq

    return padded_seq


def _unique_dates(dates: dict[str, torch.Tensor], satellites: Iterable[str]) -> torch.Tensor:
    concatenated_dates = torch.cat([dates[satellite] for satellite in satellites])
    unique_sorted_dates: torch.Tensor = torch.unique(concatenated_dates, sorted=True)
    return unique_sorted_dates


class MMapMetadata:
    """Memory map metadata class.

    Args:
        file_paths: Iterable of file paths to memory map.

    """

    def __init__(self, file_paths: Iterable[Path]) -> None:
        # init a fixed mapping between paths and mmap positions
        self.file_index_map = {file_path: idx for idx, file_path in enumerate(file_paths)}
        # init array names in mmap (assumes these are equal for all file paths)
        with np.load(next(iter(self.file_index_map.keys())), mmap_mode="r") as data:
            self.array_names = cast(list[str], data.files)

    @cached_property
    def sorted_used_files(self) -> list[str]:
        """Get list of used file paths as strings sorted by position in mmap."""
        return [
            str(file_path)
            for file_path, _ in sorted(self.file_index_map.items(), key=itemgetter(1))
        ]


class MMapStore:
    """Memory map creator class.

    Args:
        file_paths: List of paths to files to be memory mapped.
            The MMapStore instance will be cached and possibly reused
            based on (sorted) file_paths, in order to reduce opening
            too many file handles.
    """

    all_mmaps: dict[Path, dict[str, RaggedMmap]] = {}  # cache for mmaps

    def __init__(self, file_paths: Iterable[Path]) -> None:
        self.mmap_store_metadata = MMapMetadata(file_paths)

        self.mmap_data_dir: Path = (
            Settings().data_dir
            / "mmap"
            / hashlib.sha256(
                "\0".join(sorted(self.mmap_store_metadata.sorted_used_files)).encode()
            ).hexdigest()
        )

        # create a new mmap if necessary
        if not self.mmap_data_dir.is_dir():
            logger.info("No existing MMmap found! Will create it.")
            self.mmap_data_dir.mkdir(parents=True)

            for array_name in self.mmap_store_metadata.array_names:
                self._process_array_type(array_name)

        # add mmap to cache if necessary
        if self.mmap_data_dir not in self.__class__.all_mmaps:
            logger.info("Existing MMap found but not yet in memory. Will load it.")
            self.__class__.all_mmaps[self.mmap_data_dir] = {
                array_name: RaggedMmap(
                    out_dir=self.mmap_data_dir / array_name,
                    copy_before_wrapper_fn=False,
                )
                for array_name in self.mmap_store_metadata.array_names
            }

        # load mmaps from cache
        self.mmaps = self.__class__.all_mmaps[self.mmap_data_dir]

    def __getitem__(self, f: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
        return {
            array_name: {
                key: array_mmap[self.mmap_store_metadata.file_index_map[filepath]]
                for key, filepath in f.items()
            }
            for array_name, array_mmap in self.mmaps.items()
        }

    @staticmethod
    def _map_filepath_to_array(file_path: Path, array_name: str) -> np.ndarray:
        return cast(np.ndarray, np.load(file_path, mmap_mode="r")[array_name])

    def _process_array_type(self, array_name: str) -> None:
        RaggedMmap.from_generator(
            out_dir=self.mmap_data_dir / array_name,
            sample_generator=map(
                partial(self._map_filepath_to_array, array_name=array_name),
                self.mmap_store_metadata.sorted_used_files,
            ),
            batch_size=1024,
            verbose=True,
        )
