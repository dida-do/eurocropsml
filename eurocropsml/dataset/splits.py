"""Generating region based EuroCrops dataset splits."""

import json
import logging
import shutil
from pathlib import Path
from typing import Literal, cast

from sklearn.model_selection import train_test_split

from eurocropsml.dataset.config import EuroCropsSplit
from eurocropsml.dataset.download import _download_file, _get_zenodo_record
from eurocropsml.dataset.utils import (
    _create_final_dict,
    _create_finetune_set,
    _downsample_class,
    _filter_regions,
    _order_classes,
    _save_counts_to_csv,
    _save_to_dict,
    _save_to_json,
    _split_dataset,
    read_files,
)
from eurocropsml.utils import _unzip_file

logger = logging.getLogger(__name__)


def get_split_dir(split_dir: Path, split_name: str) -> Path:
    """Get directory where splits are saved.

    Args:
        split_dir: Path for the splits root directory.
        split_name: Subdirectory name for named split.

    Returns:
        The full path, consisting of root directory and subdirectory.
    """
    return split_dir.joinpath("split", split_name)


def create_splits(
    split_config: EuroCropsSplit,
    split_dir: Path,
    download_url: str | None = None,
) -> None:
    """Create EuroCrops dataset splits.

    Args:
        split_config: Configuration used for splitting dataset.
        split_dir: Data directory where split folder is saved.
        download_url: Zenodo download URL. Used for downloading benchmark split.

    Raises:
        ValueError: If download_url not provided and benchmark set to True.
    """
    if download_url is None and split_config.benchmark is True:
        raise ValueError(
            "A Zenodo download URL is needed if benchmark is set to True."
            "Either provide the URL or set benchmark to False."
        )
    split_dir = get_split_dir(split_dir, split_config.base_name)
    splits = split_config.pretrain_classes
    meadow_class = split_config.meadow_class
    for split in splits:
        _build_dataset_split(
            data_dir=split_config.data_dir,
            split=split,  # type: ignore[arg-type]
            satellite=split_config.satellite,
            year=str(split_config.year),
            split_dir=split_dir,
            zenodo_base_url=download_url,
            pretrain_classes=set(split_config.pretrain_classes[split]),
            finetune_classes=(
                finetune_classes
                if (finetune_classes := split_config.finetune_classes.get(split)) is None
                else set(finetune_classes)
            ),
            pretrain_regions=set(split_config.pretrain_regions),
            finetune_regions=set(split_config.finetune_regions),
            num_samples=split_config.num_samples,
            meadow_class=meadow_class,
            force_rebuild=False,
            seed=split_config.random_seed,
            benchmark=split_config.benchmark,
        )


def _build_dataset_split(
    data_dir: Path,
    split_dir: Path,
    split: Literal["class", "regionclass", "region"],
    satellite: list[Literal["S1", "S2"]],
    year: str,
    num_samples: dict[str, str | int | list[int | str]],
    seed: int,
    pretrain_classes: set,
    finetune_classes: set | None = None,
    pretrain_regions: set | None = None,
    finetune_regions: set | None = None,
    meadow_class: int | None = None,
    force_rebuild: bool = False,
    benchmark: bool = False,
    zenodo_base_url: str | None = None,
) -> None:
    """Build data split for EuroCrops data.

    Args:
        data_dir: Directory where labels and data are stored.
        split_dir: Directory where split file is going to be saved to.
        split: Kind of data split to apply.
        satellite: Whether to build the splits using Sentinel-1 or Sentinel-2 or both.
        year: Year for which data are to be processed.
        num_samples: Number of samples to sample for finetuning.
        seed: Randoms seed,
        pretrain_classes: Classes of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_classes: Classes of the requested dataset split for finetuning.
        pretrain_regions: Regions of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_regions: Regions of the requested dataset split for finetuning.
            None if EuroCrops should only be used for pretraining.
        meadow_class: Meadow class identifier. If specified, for the pre-training split,
            the meadow class will be downsampled to the median frequency of all other classes
            If None, no downsampling is taking place.
        force_rebuild: Whether to rebuild split if split file already exists.
        benchmark: Flag in order to build the same split as used in the EuroCropsML dataset
            (https://arxiv.org/abs/2407.17458). The split was created when only Sentinel-2 data
            was available. If `benchmark` is set to True, the split will be loaded from Zenodo
            version 9. If 'S1' in `satellite` and the `EuroCropsSplit.base_name` matches one of the
            Zenodo splits, then for pre-training the remaining Sentinel-1 parcels (which are
            not in the S2 data) are distributed between train and validation. For fine-tuning,
            there are only 149 parcels in S1 which are not in S2. We therefore ignore them
            completely, s.t. the fine-tuning split remains exactly the same. If the `benchmark` is
            set to False, a new deterministic train-val(-test) split is created based on all
            parcels present in the data.
            If `split`!="region", `benchmark` is set to False.
        zenodo_base_url: Base url for downloading benchmark region-split from Zenodo.

    Raises:
        FileNotFoundError: If `data_dir` is not a directory.
        ValueError: If `pretrain_regions` is not specified but we want to split by regions.
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(str(data_dir) + " is not a directory.")

    if not force_rebuild:
        split_files = [
            split_dir.joinpath("pretrain", f"{split}_split.json"),
            split_dir.joinpath("meta", f"{split}_split.json"),
        ]
        for num in num_samples["train"]:  # type: ignore[union-attr]
            split_files.append(split_dir.joinpath("finetune", f"{split}_split_{num}.json"))

        if all(file.is_file() for file in split_files):
            logger.info(
                "Files already exist and force_rebuild=False. "
                f"Skipping recreation of {split}-split.",
            )
            return
    logger.info(f"Creating {split}-split...")
    if split != "region" and benchmark is True:
        benchmark = False
        logger.info(
            "Basing the split only on Sentinel-2 for creating the benchmark dataset "
            "is only possible for split='region'. Setting benchmark to False."
        )
    if split == "class":
        split_dataset_by_class(
            data_dir,
            split_dir,
            satellite,
            year=year,
            num_samples=num_samples,
            pretrain_classes=pretrain_classes,
            finetune_classes=finetune_classes,
            meadow_class=meadow_class,
            seed=seed,
        )
    else:
        if pretrain_regions is None:
            raise ValueError("Please specify the relevant pretrain regions to sample from.")
        split_dataset_by_region(
            data_dir,
            split_dir,
            split,
            satellite,
            year=year,
            num_samples=num_samples,
            pretrain_classes=pretrain_classes,
            finetune_classes=finetune_classes,
            pretrain_regions=pretrain_regions,
            finetune_regions=finetune_regions,
            meadow_class=meadow_class,
            seed=seed,
            benchmark=benchmark,
            zenodo_base_url=cast(str, zenodo_base_url),
        )


def split_dataset_by_class(
    data_dir: Path,
    split_dir: Path,
    satellite: list[Literal["S1", "S2"]],
    year: str,
    num_samples: dict[str, str | int | list[int | str]],
    seed: int,
    pretrain_classes: set[int],
    finetune_classes: set[int] | None = None,
    meadow_class: int | None = None,
    test_size: float = 0.2,
) -> None:
    """Split dataset by classes.

    Args:
        data_dir: Path that contains `.npy` files where labels and data are stored.
        split_dir: Directory where splits are going to be saved to.
        satellite: Whether to build the splits using Sentinel-1 or Sentinel-2 or both.
        year: Year for which data are to be processed.
        num_samples: Number of samples to sample for finetuning.
        seed: Random seed for data split.
        pretrain_classes: List with classes used for filtering the data.
        finetune_classes: List with classes used for filtering the data.
        meadow_class: Meadow class identifier. If specified, for the pre-training split,
            the meadow class will be downsampled to the median frequency of all other classes
            If None, no downsampling is taking place.
        test_size: Amount of data used for validation (test set).
            Defaults to 0.2.

    Raises:
        Exception: If there are similar samples within pretrain and finetune data-split.

    """

    # split into pretrain and finetune dataset
    pretrain_dataset, finetune_dataset = _split_dataset(
        data_dir=data_dir,
        satellite=satellite,
        year=year,
        pretrain_classes=pretrain_classes,
        finetune_classes=finetune_classes,
    )

    if meadow_class is not None:
        pretrain_list: list[str] = _downsample_class(
            pretrain_dataset, seed=seed, class_key=meadow_class
        )
    else:
        pretrain_list = [file for files in pretrain_dataset.values() for file in files]

    # save finetuning split
    if finetune_dataset is not None:
        _create_finetune_set(
            finetune_dataset,
            split_dir.joinpath("finetune"),
            "class",
            pretrain_list,
            num_samples,
            test_size,
            seed,
        )

    # sorting list to make train_test_split deterministic
    pretrain_list.sort()
    # save pretraining split
    train, val = train_test_split(pretrain_list, test_size=test_size, random_state=seed)

    pretrain_dict = _save_to_dict(train, val)

    _save_to_json(split_dir.joinpath("pretrain", "class_split.json"), pretrain_dict)

    metatrain_dict = _order_classes(train)
    metaval_dict = _order_classes(val)

    meta_dict: dict = {"train": metatrain_dict, "val": metaval_dict}

    _save_to_json(split_dir.joinpath("meta", "class_split.json"), meta_dict)

    _save_counts_to_csv(pretrain_list, split_dir.joinpath("counts", "pretrain"), "class")


def split_dataset_by_region(
    data_dir: Path,
    split_dir: Path,
    split: Literal["region", "regionclass"],
    satellite: list[Literal["S1", "S2"]],
    year: str,
    num_samples: dict[str, str | int | list[int | str]],
    seed: int,
    benchmark: bool,
    zenodo_base_url: str,
    pretrain_classes: set[int],
    pretrain_regions: set[str],
    finetune_classes: set[int] | None = None,
    finetune_regions: set[str] | None = None,
    meadow_class: int | None = None,
    test_size: float = 0.2,
) -> None:
    """Split dataset by regions or regions and classes.

    Args:
        data_dir: Path that contains `.npy` files where labels and data are stored.
        split_dir: Directory where splits are going to be saved to.
        split: Kind of data split to apply.
        satellite: Whether to build the splits using Sentinel-1 or Sentinel-2 or both.
        year: Year for which data are to be processed.
        num_samples: Number of samples to sample for finetuning.
        seed: Random seed for data split.
        benchmark: Flag in order to build the same split as used in the EuroCropsML dataset
            (https://arxiv.org/abs/2407.17458). The split was created when only Sentinel-2 data
            was available. If `benchmark` is set to True, the split will be loaded from Zenodo
            version 9. If 'S1' in `satellite` and the `EuroCropsSplit.base_name` matches one of the
            Zenodo splits, then for pre-training the remaining Sentinel-1 parcels (which are
            not in the S2 data) are distributed between train and validation. For fine-tuning,
            there are only 149 parcels in S1 which are not in S2. We therefore ignore them
            completely, s.t. the fine-tuning split remains exactly the same. If the `benchmark` is
            set to False, a new deterministic train-val(-test) split is created based on all
            parcels present in the data.
            If `split`!="region", `benchmark` is set to False.
        zenodo_base_url: Base url for downloading benchmark region-split from Zenodo (version 9).
        pretrain_classes: Classes of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_classes: Classes of the requested dataset split for finetuning.
        pretrain_regions: Regions of the requested dataset split for
            hyperparameter tuning and pretraining.
        finetune_regions: Regions of the requested dataset split for finetuning.
            None if EuroCrops should only be used for pretraining.
        meadow_class: Meadow class identifier. If specified, for the pre-training split,
            the meadow class will be downsampled to the median frequency of all other classes
            If None, no downsampling is taking place.
        test_size: Amount of data used for validation (test set).
            Defaults to 0.2.

    Raises:
        Exception: If there are similar samples within pretrain and finetune data-split.
    """

    regions = (
        pretrain_regions | finetune_regions if finetune_regions is not None else pretrain_regions
    )

    classes = (
        pretrain_classes | finetune_classes
        if finetune_classes is not None and split == "region"
        else pretrain_classes
    )

    create_pretrain: bool = True
    if benchmark is True:
        create_pretrain = _get_benchmark_dataset(
            data_dir,
            split_dir.parent.parent,
            split_dir,
            zenodo_base_url,
            satellite,
            pretrain_regions,
            test_size=test_size,
            seed=seed,
        )

        if create_pretrain is True:
            # only fine-tuning split was copied for benchmarking
            # create new pre-training split
            finetune_classes = None
            finetune_regions = None

    if create_pretrain is True:

        # split into pretrain and finetune dataset
        pretrain_dataset, finetune_dataset = _split_dataset(
            data_dir=data_dir,
            satellite=satellite,
            year=year,
            pretrain_classes=classes,
            finetune_classes=finetune_classes if split == "regionclass" else None,
            regions=set(regions),
        )

        if split == "region":
            finetune_dataset = pretrain_dataset.copy()

        pretrain_dataset = _filter_regions(pretrain_dataset, pretrain_regions)

        if meadow_class is not None and meadow_class in pretrain_classes:
            pretrain_list: list[str] = _downsample_class(
                pretrain_dataset, seed=seed, class_key=meadow_class
            )
        else:
            pretrain_list = [file for files in pretrain_dataset.values() for file in files]

        if (
            finetune_dataset is not None and finetune_regions is not None
        ):  # otherwise EuroCrops is solely used for pretraining
            finetune_dataset = _filter_regions(finetune_dataset, finetune_regions)

            _create_finetune_set(
                finetune_dataset,
                split_dir.joinpath("finetune"),
                split,
                pretrain_list,
                num_samples,
                test_size,
                seed,
            )

        # sorting list to make train_test_split deterministic
        pretrain_list.sort()
        # save pretraining split
        train, val = train_test_split(pretrain_list, test_size=test_size, random_state=seed)

        pretrain_dict = _save_to_dict(train, val)

        _save_to_json(split_dir.joinpath("pretrain", f"{split}_split.json"), pretrain_dict)

        _save_counts_to_csv(pretrain_list, split_dir.joinpath("counts", "pretrain"), split)

        meta_dict: dict = {
            "train": _create_final_dict(train, pretrain_regions),
            "val": _create_final_dict(val, pretrain_regions),
        }

        _save_to_json(split_dir.joinpath("meta", f"{split}_split.json"), meta_dict)


def _get_benchmark_dataset(
    data_dir: Path,
    parent_split_dir: Path,
    split_dir: Path,
    zenodo_base_url: str,
    satellite: list[Literal["S1", "S2"]],
    pretrain_regions: set[str],
    split: Literal["region"] = "region",
    test_size: float = 0.2,
    seed: int = 42,
) -> bool:

    logger.info("Benchmark set to True. Downloading benchmark split from Zenodo version 11...")
    selected_version = cast(dict, _get_zenodo_record(zenodo_base_url, version_number=11))
    for file_entry in selected_version["files"]:
        if file_entry["key"] == "split.zip":
            file_url: str = file_entry["links"]["self"]
            local_path: Path = parent_split_dir.joinpath("split.zip")
            _download_file("split.zip", file_url, local_path, file_entry.get("checksum", ""))
            logger.info(f"Unzipping {local_path}...")
            _unzip_file(local_path, parent_split_dir)
            break

    # if split equals one of the splits from Zenodo, we simply add S1 data (if requested) to
    # the existing pre-training split from Zenodo
    if split_dir.name in [
        "latvia_vs_estonia",
        "latvia_portugal_vs_estonia",
        "overlap_latvia_vs_estona",
        "overlap_latvia_portugal_vs_estonia",
    ]:
        if "S1" in satellite:
            filtered_s1: list[str] = []
            s1_files: set = read_files(data_dir.joinpath("S1"))
            with open(split_dir.joinpath("pretrain", "region_split.json"), "r") as file:
                pretrain_dataset = json.load(file)
            train = pretrain_dataset["train"]
            val = pretrain_dataset["val"]
            # get the files that are not present in S2 (only for pretraining)
            # filter by regions
            filtered_s1 = [
                file
                for file in s1_files
                if file not in pretrain_dataset.values()
                and file.startswith(tuple(pretrain_regions))
            ]
            if not filtered_s1:
                logger.info(
                    f"Split {split_dir.name} was downloaded from Zenodo and no additional "
                    "Sentinel-1 pre-training data found. Pre-training and fine-tuning split are "
                    "copied from Zenodo for benchmarking."
                )
            else:
                logger.info(
                    f"Split {split_dir.name} was downloaded from Zenodo. Additional "
                    "Sentinel-1 pre-training data will be added. Fine-tuning split is copied from "
                    "Zenodo for benchmarking."
                )
                # sorting list to make train_test_split deterministic
                filtered_s1.sort()
                s1_train, s1_val = train_test_split(
                    filtered_s1, test_size=test_size, random_state=seed
                )
                train.extend(s1_train)
                val.extend(s1_val)
                # sort train and validation list s.t. S1 data is not just appended at the end
                train.sort()
                val.sort()

                pretrain_dict = _save_to_dict(train, val)

                pretrain_list = list(pretrain_dict.values())

                _save_to_json(split_dir.joinpath("pretrain", f"{split}_split.json"), pretrain_dict)

                _save_counts_to_csv(pretrain_list, split_dir.joinpath("counts", "pretrain"), split)

                meta_dict: dict = {
                    "train": _create_final_dict(train, pretrain_regions),
                    "val": _create_final_dict(val, pretrain_regions),
                }

                _save_to_json(split_dir.joinpath("meta", f"{split}_split.json"), meta_dict)

        else:
            logger.info(f"Split {split_dir.name} was downloaded from Zenodo. Nothing else to do.")
        return False

    # if split is different, meaning the pre-training data is different, we only copy the
    # fine-tuning split for benchmarking and create a new pre-training split
    else:
        logger.info(
            f"Split {split_dir.name} does not exist, yet. A new pre-training split is "
            "created. The fine-tuning split is copied from Zenodo for benchmarking."
        )
        finetune_dir: Path = parent_split_dir.joinpath("split", "latvia_vs_estonia", "finetune")
        shutil.copytree(finetune_dir, split_dir.joinpath("finetune"))

        return True
