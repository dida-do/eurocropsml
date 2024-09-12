"""Generating region based EuroCrops dataset splits."""

import logging
from pathlib import Path
from typing import Literal

from eurocropsml.dataset.config import EuroCropsSplit
from eurocropsml.dataset.utils import split_dataset_by_class, split_dataset_by_region

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


def create_splits(split_config: EuroCropsSplit, split_dir: Path) -> None:
    """Create EuroCrops dataset splits.

    Args:
        split_config: Configuration used for splitting dataset.
        split_dir: Data directory where split folder is saved.
    """
    split_dir = get_split_dir(split_dir, split_config.base_name)
    splits = split_config.pretrain_classes
    meadow_class = split_config.meadow_class
    for split in splits:
        _build_dataset_split(
            data_dir=split_config.data_dir,
            split=split,  # type: ignore[arg-type]
            satellite=split_config.satellite,
            split_dir=split_dir,
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
    num_samples: dict[str, str | int | list[int | str]],
    seed: int,
    pretrain_classes: set,
    finetune_classes: set | None = None,
    pretrain_regions: set | None = None,
    finetune_regions: set | None = None,
    meadow_class: int | None = None,
    force_rebuild: bool = False,
    benchmark: bool = False,
) -> None:
    """Build data split for EuroCrops data.

    Args:
        data_dir: Directory where labels and data are stored.
        split_dir: Directory where split file is going to be saved to.
        split: Kind of data split to apply.
        satellite: Whether to build the splits using Sentinel-1 or Sentinel-2 or both.
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
            was available. If benchmark is set to True and the 'S1' in satellite is selected, the
            split will be created using only S2 data. For pre-training, the remaining Sentinel-1
            parcels (that are not present in the S2 data) will then be distributed between train
            and validation. For fine-tuning, there are only 149 parcels in S1 which are not in S2.
            We therefore neglect these completely, s.t. the fine-tuning split stays compltely the
            same. If the benchmark is set to False, a new train-val-test split will be created
            based on all parcels present in the data.

    Raises:
        FileNotFoundError: If data_dir is not a directory.
        ValueError: If regions is not specified but we want to split by regions.
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
                "Files already exist. Skipping recreation.",
            )
            return
    if split == "class":
        split_dataset_by_class(
            data_dir,
            split_dir,
            satellite,
            num_samples=num_samples,
            pretrain_classes=pretrain_classes,
            finetune_classes=finetune_classes,
            meadow_class=meadow_class,
            seed=seed,
        )
    else:
        if pretrain_regions is None:
            raise ValueError("Please specify the relevant pretrain regions to sample from.")
        if split == "regionclass" and benchmark is True:
            benchmark = False
            logger.info(
                "Basing the split only on Sentinel-2 for creating the benchmark dataset "
                "is only possible for split='region'. Setting benchmark to False."
            )
        split_dataset_by_region(
            data_dir,
            split_dir,
            split,
            satellite,
            num_samples=num_samples,
            pretrain_classes=pretrain_classes,
            finetune_classes=finetune_classes,
            pretrain_regions=pretrain_regions,
            finetune_regions=finetune_regions,
            meadow_class=meadow_class,
            seed=seed,
            benchmark=benchmark,
        )
