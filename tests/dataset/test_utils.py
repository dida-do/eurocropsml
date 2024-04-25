from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from eurocropsml.dataset.config import EuroCropsSplit
from eurocropsml.dataset.utils import (
    _create_final_dict,
    _downsample_class,
    _filter_classes,
    _filter_regions,
    _order_classes,
    _sample_max_samples,
    pad_seq_to_366,
)


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


@pytest.fixture
def test_split_config() -> EuroCropsSplit:
    config = EuroCropsSplit(
        base_name="test",
        data_dir=Path("test_dir"),
        random_seed=42,
        num_samples={"train": 1, "validation": 2, "test": 2},
        meadow_class=102,
        pretrain_classes={
            "class": [102, 103, 107],
            "region": [102, 103, 107],
            "regionclass": [102, 103, 107],
        },
        finetune_classes={
            "class": [105, 104, 108],
            "region": [102, 103, 107],
            "regionclass": [105, 104, 108],
        },
        pretrain_regions=["PT1", "PT2"],
        finetune_regions=["FT1", "FT2"],
    )

    return config


@pytest.fixture
def test_data() -> list[str]:
    return [
        "PT1_parcel1_102.npz",
        "PT1_parcel2_103.npz",
        "PT1_parcel3_107.npz",
        "PT2_parcel4_107.npz",
        "PT2_parcel5_102.npz",
        "PT2_parcel6_107.npz",
        "PT2_parcel7_103.npz",
        "PT2_parcel22_107.npz",
        "PT1_parcel25_105.npz",
        "PT1_parcel26_103.npz",
        "PT1_parcel27_107.npz",
        "PT1_parcel28_103.npz",
        "PT2_parcel29_103.npz",
        "PT2_parcel5_102.npz",
        "FT1_parcel8_105.npz",
        "FT1_parcel9_104.npz",
        "FT1_parcel10_108.npz",
        "FT2_parcel11_105.npz",
        "FT2_parcel12_104.npz",
        "FT2_parcel13_108.npz",
        "FT1_parcel14_102.npz",
        "FT1_parcel15_103.npz",
        "FT1_parcel16_107.npz",
        "FT2_parcel17_102.npz",
        "FT2_parcel18_103.npz",
        "FT2_parcel19_107.npz",
        "FT1_parcel20_103.npz",
        "FT2_parcel21_105.npz",
        "FT2_parcel24_102.npz",
    ]


def test_pad_seq_to_366(test_arrays: tuple[np.ndarray, torch.Tensor]) -> None:
    test_data, test_dates = test_arrays
    np_data = pad_seq_to_366(test_data, test_dates)

    assert np_data.shape == (366, test_data.shape[1])


def test_split_dataset_by_region(test_split_config: EuroCropsSplit, test_data: list) -> None:
    split = "region"
    _split_dataset_by_region(split, test_split_config, test_data)


def test_split_dataset_by_regionclass(test_split_config: EuroCropsSplit, test_data: list) -> None:
    split = "regionclass"
    _split_dataset_by_region(split, test_split_config, test_data)


def test_split_dataset_by_class(test_split_config: EuroCropsSplit, test_data: list) -> None:
    pretrain_classes: set = set(test_split_config.pretrain_classes["class"])
    finetune_classes: set = set(test_split_config.finetune_classes["class"])
    pretrain_dataset, finetune_dataset = _filter_classes(
        test_data,
        pretrain_classes,
        finetune_classes,
    )

    if test_split_config.meadow_class is not None:
        pretrain_list: list[str] = _downsample_class(
            pretrain_dataset,
            seed=test_split_config.random_seed,
            class_key=test_split_config.meadow_class,
        )
    else:
        pretrain_list = [file for files in pretrain_dataset.values() for file in files]

    # save finetuning split
    if finetune_dataset is not None:
        _create_finetune_set(
            finetune_dataset,
            pretrain_list,
            test_split_config.num_samples,
            0.2,
            test_split_config.random_seed,
        )

    # save pretraining split
    train, val = train_test_split(
        pretrain_list, test_size=0.2, random_state=test_split_config.random_seed
    )

    metatrain_dict = _order_classes(train)
    metaval_dict = _order_classes(val)

    assert len(metatrain_dict) == len(pretrain_classes)
    assert len(metaval_dict) == len(pretrain_classes)


def _split_dataset_by_region(
    split: str, test_split_config: EuroCropsSplit, test_data: list
) -> None:

    regions: set = (
        set(test_split_config.pretrain_regions) | set(test_split_config.finetune_regions)
        if test_split_config.finetune_regions is not None
        else set(test_split_config.pretrain_regions)
    )

    pretrain_classes: set = set(test_split_config.pretrain_classes[split])
    finetune_classes: set = set(test_split_config.finetune_classes[split])

    classes = (
        pretrain_classes | finetune_classes
        if finetune_classes is not None and split == "region"
        else set(pretrain_classes)
    )

    # pre-filter by regions for region and regionclass split
    if regions is not None:
        files_list = [file for file in test_data if file.startswith(tuple(regions))]

    pretrain_dataset, finetune_dataset = _filter_classes(
        files_list,
        classes,
        finetune_classes=finetune_classes if split == "regionclass" else None,
    )

    if split == "region":
        finetune_dataset = pretrain_dataset.copy()

    pretrain_dataset = _filter_regions(pretrain_dataset, set(test_split_config.pretrain_regions))

    if (
        test_split_config.meadow_class is not None
        and test_split_config.meadow_class in pretrain_classes
    ):
        pretrain_list: list[str] = _downsample_class(
            pretrain_dataset,
            seed=test_split_config.random_seed,
            class_key=test_split_config.meadow_class,
        )
    else:
        pretrain_list = [file for files in pretrain_dataset.values() for file in files]

    if (
        finetune_dataset is not None and test_split_config.finetune_regions is not None
    ):  # otherwise EuroCrops is solely used for pretraining
        finetune_dataset = _filter_regions(
            finetune_dataset, set(test_split_config.finetune_regions)
        )

        _create_finetune_set(
            finetune_dataset,
            pretrain_list,
            test_split_config.num_samples,
            0.2,
            test_split_config.random_seed,
        )

    train, val = train_test_split(
        pretrain_list, test_size=0.2, random_state=test_split_config.random_seed
    )

    meta_dict: dict = {
        "train": _create_final_dict(train, set(test_split_config.pretrain_regions)),
        "val": _create_final_dict(val, set(test_split_config.pretrain_regions)),
    }

    assert len(meta_dict["train"]) == len(test_split_config.pretrain_regions)
    assert len(meta_dict["val"]) == len(test_split_config.pretrain_regions)


def _get_classes(dataset: list[str]) -> dict[str, list[str]]:
    dataset_dict = defaultdict(list)
    for val in dataset:
        c = val.split(".")[0].split("_")[-1]
        dataset_dict[c].append(val)

    return dataset_dict


def _create_finetune_set(
    finetune_dataset: dict[int, list[str]],
    pretrain_list: list[str],
    num_samples: dict[str, str | int | list[int | str]],
    test_size: float,
    seed: int,
) -> None:
    finetune_list: list[str] = [
        value for values_list in finetune_dataset.values() for value in values_list
    ]

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

    assert len(finetune_train) + len(finetune_val) + len(finetune_test) == len(finetune_list)

    sample_list: list[str | int]
    if isinstance(num_samples["train"], list):
        sample_list = num_samples["train"]
    else:
        sample_list = [cast(int, num_samples["train"])]

    for max_samples in sample_list:
        train = _sample_max_samples(finetune_train, int(max_samples), seed)

    dataset_dict = _get_classes(finetune_train)
    assert int(max_samples) * len(dataset_dict.keys()) == len(train)
