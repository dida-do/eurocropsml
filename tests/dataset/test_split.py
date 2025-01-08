import json
import random
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from eurocropsml.dataset.config import EuroCropsSplit
from eurocropsml.dataset.splits import create_splits, get_split_dir


@pytest.fixture
def test_split_config() -> tuple[tempfile.TemporaryDirectory, EuroCropsSplit]:
    temp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()

    config = EuroCropsSplit(
        base_name="test",
        data_dir=Path(temp_dir.name),
        random_seed=42,
        num_samples={"train": [1], "validation": 2, "test": 2},
        satellie=["S1", "S2"],
        pretrain_classes={
            "class": [11102, 11103, 11107],
            "region": [11102, 11103, 11104, 11105, 11107, 11108],
            "regionclass": [11102, 11103, 11104],
        },
        finetune_classes={
            "class": [11105, 11104, 11108],
            "region": [11102, 11103, 11104, 11105, 11107, 11108],
            "regionclass": [11105, 11107, 11108],
        },
        pretrain_regions=["PT1", "PT2"],
        finetune_regions=["FT1", "FT2"],
    )

    return temp_dir, config


@pytest.fixture
def test_data() -> list[str]:
    random.seed(42)
    file_list: list[str] = []

    for i in range(501, 1501):
        region = random.choice(["PT1", "PT2"])
        c = random.choice([11102, 11103, 11104, 11105, 11107, 11108])
        file_list.append(f"{region}_parcel{i}_{c}.npz")

    for i in range(1501, 2001):
        region = random.choice(["FT1", "FT2"])
        c = random.choice([11102, 11103, 11104, 11105, 11107, 11108])
        file_list.append(f"{region}_parcel{i}_{c}.npz")

    return file_list


@pytest.fixture
def test_split_dir() -> tuple[tempfile.TemporaryDirectory, Path]:
    temp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()

    return temp_dir, Path(temp_dir.name)


def _flatten_values(dic: dict[str, dict]) -> list:
    return [
        value
        for v in dic.values()
        for value in (_flatten_values(v) if isinstance(v, dict) else [v])
    ]


def test_create_splits(
    test_data: list[str],
    test_split_config: tuple[tempfile.TemporaryDirectory, EuroCropsSplit],
    test_split_dir: tuple[tempfile.TemporaryDirectory, Path],
) -> None:
    temp_data_dir = test_split_config[0]
    config = test_split_config[1]

    temp_split_dir = test_split_dir[0]
    split_dir = test_split_dir[1]

    # create subdirs
    for satellite in config.satellite:
        satellite_path: Path = config.data_dir / satellite / str(config.year)
        satellite_path.mkdir(exist_ok=True, parents=True)
        for file_name in test_data:
            file_path = satellite_path / file_name
            array1 = np.random.rand(5)
            array2 = np.random.rand(3, 3)
            np.savez(file_path, array1=array1, array2=array2)

    create_splits(config, split_dir)

    final_split_dir: Path = get_split_dir(split_dir, config.base_name)

    for folder in final_split_dir.iterdir():
        if folder.name != "counts":
            files = list(folder.iterdir())

            if folder.name in ["pretrain", "meta"]:
                assert len(files) == len(config.pretrain_classes)
                for file in files:
                    with open(file, "r") as split_file:
                        split_data: dict[str, list[str]] = json.load(split_file)
                        split: str = file.stem
                        split = split.split("_")[0]
                        if folder.name == "meta":
                            train: list = sum(_flatten_values(cast(dict, split_data["train"])), [])
                            val: list = sum(_flatten_values(cast(dict, split_data["val"])), [])
                        else:
                            train = split_data["train"]
                            val = split_data["val"]
                        match split:
                            case "class":
                                data: list[Path] | list[str] = [
                                    file
                                    for file in test_data
                                    if Path(file).stem.endswith(
                                        tuple(str(c) for c in config.pretrain_classes[split])
                                    )
                                ]
                                if folder.name == "meta":
                                    assert len(split_data["train"]) == len(
                                        config.pretrain_classes[split]
                                    )
                                    assert len(split_data["val"]) == len(
                                        config.pretrain_classes[split]
                                    )
                                assert len(train) + len(val) == len(data)
                            case "regionclass":
                                data = [
                                    Path(file)
                                    for file in test_data
                                    if file.startswith(tuple(config.pretrain_regions))
                                ]
                                data = [
                                    file
                                    for file in data
                                    if file.stem.endswith(
                                        tuple(str(c) for c in config.pretrain_classes[split])
                                    )
                                ]
                                assert len(train) + len(val) == len(data)
                            case "region":
                                data = [
                                    file
                                    for file in test_data
                                    if file.startswith(tuple(config.pretrain_regions))
                                ]
                                if folder.name == "meta":
                                    assert len(split_data["train"]) == len(config.pretrain_regions)
                                    assert len(split_data["val"]) == len(config.pretrain_regions)

                                assert len(train) + len(val) == len(data)

            else:
                max_samples: list[int] = cast(list, config.num_samples["train"])
                assert len(files) == len(max_samples) * len(config.pretrain_classes)
                for file in files:
                    with open(file, "r") as split_file:
                        split_data = json.load(split_file)

                        split = file.stem
                        split = split.split("_")[0]

                        assert len(split_data["val"]) == config.num_samples["validation"]
                        assert len(split_data["test"]) == config.num_samples["test"]
                        if split == "region":
                            data = [
                                Path(file)
                                for file in test_data
                                if file.startswith(tuple(config.finetune_regions))
                            ]
                            classes: set = {file.stem.split("_")[-1] for file in data}
                            assert len(split_data["train"]) in [
                                len(classes) * sample for sample in max_samples
                            ]
                        else:
                            assert len(split_data["train"]) in [
                                len(config.finetune_classes[split]) * sample
                                for sample in max_samples
                            ]

    temp_data_dir.cleanup()
    temp_split_dir.cleanup()
