"""Handling configurations for the dataset creation."""

import logging
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, field_validator

from eurocropsml.acquisition.config import S1_BANDS, S2_BANDS
from eurocropsml.settings import Settings

logger = logging.getLogger(__name__)


class EuroCropsDatasetPreprocessConfig(BaseModel):
    """Configuration for downloading and preprocessing EuroCrops dataset.

    Args:
        download_url: Zenodo URL do download dataset from.
        raw_data_dir: Directory where the raw EuroCropsML data is stored (from data acquisiton).
        preprocess_dir: Directory where the preprocessed data is stored.
        band4_t1: Lower threshold for band 4 for identifying cloudy pixels.
        band4_t2: Upper threshold for band 4 for identifying cloudy pixels.
        band4_prob_threshold: Probability threshold for filtering clouds that decides whether an
            observation is defined cloudy or non-cloudy.
        filter_clouds: Whether to filter clouds from Sentinel-2 time series.
        num_workers: Number of workers used during multiprocessing.
        excl_classes: Classes that should be excluded even before preprocessing.
        keep_classes: Classes to keep for preprocessing. This comes in handy if for example only
            a couple of classes are relevant. In that case, it massively speeds up the pre-
            processing.
        satellite: Preprocess Sentinel-1 or Sentinel-2.
        bands: If this is None, the default bands stated in the global variables will be used.
            These are also the ones available in the ready-to-use EuroCropsML dataset.
            If during your own data acquisition not all bands or different bands were acquired,
            please define them here.
        year: Year for which data are to be processed.

    """

    download_url: str = "https://zenodo.org/api/records/10629610/versions"
    raw_data_dir: Path
    preprocess_dir: Path
    band4_t1: float = 0.07
    band4_t2: float = 0.25
    band4_prob_threshold: float = 0.5
    filter_clouds: bool = True
    num_workers: int | None = None
    excl_classes: list[int] = []
    keep_classes: list[int] = []
    satellite: Literal["S1", "S2"] = "S2"
    bands: list[str] | None = None
    year: int = 2021

    @field_validator("raw_data_dir", "preprocess_dir")
    @classmethod
    def relative_path(cls, v: Path) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        if not v.is_absolute():
            v = Settings().data_dir.joinpath(v)
        return v


class EuroCropsSplit(BaseModel):
    """Configuration for building EuroCrops splits."""

    base_name: str
    data_dir: Path
    random_seed: int
    num_samples: dict[str, str | int | list[int | str]]

    meadow_class: int | None = None

    satellite: list[Literal["S1", "S2"]] = ["S2"]

    year: int = 2021
    benchmark: bool = False

    pretrain_classes: dict[str, list[int]]
    finetune_classes: dict[str, list[int]] = {}

    pretrain_regions: list[str]
    finetune_regions: list[str] = []

    @field_validator("data_dir")
    @classmethod
    def relative_path(cls, v: Path) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        if not v.is_absolute():
            v = Settings().data_dir.joinpath(v)
        return v


class EuroCropsConfig(BaseModel):
    """Main configuration for building EuroCrops splits."""

    preprocess: EuroCropsDatasetPreprocessConfig
    split: EuroCropsSplit


class EuroCropsDatasetConfig(BaseModel):
    """Configuration for the EuroCrops dataset."""

    remove_s2_bands: list[str] | None = None
    date_type: Literal["day", "month"] = "day"
    filter_clouds: bool = True
    normalize: bool = True
    data_sources: list[Literal["S1", "S2"]] = ["S2"]
    s1_bands: list[str] | None = None
    s2_bands: list[str] | None = None
    year: int = 2021
    # max_samples corresponds to maximum number of samples per class for finetune training data.
    # If ["all"], all samples are used, if e.g. [1, 2, "all"], three use-cases are created where
    # number of samples per class are 1 or 2 respectively, or where all samples are used.
    max_samples: list[int | str] = ["all"]
    metrics: list[str] = ["Acc"]

    split: Literal["class", "regionclass", "region"]

    total_num_channels: int = 0

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.post_init()

    def post_init(self) -> None:
        """Make dynamic config based on initialized params."""
        num_channels: int = 0
        if "S1" in self.data_sources:
            if self.s1_bands is None:
                self.s1_bands = cast(list, S1_BANDS)
                logger.info(f"No S1 bands defined. Setting to default {', '.join(S1_BANDS)}.")
            num_channels += len(self.s1_bands)
        if "S2" in self.data_sources:
            if self.s2_bands is None:
                self.s2_bands = cast(list, S2_BANDS)
                logger.info(f"No S2 bands defined. Setting to default {', '.join(S2_BANDS)}.")
            self.s2_bands = cast(list, self.s2_bands)
            num_channels += len(self.s2_bands)
            if self.remove_s2_bands is not None:
                self.remove_s2_bands = cast(list, self.remove_s2_bands)
                if any(item not in self.s2_bands for item in self.remove_s2_bands):
                    logger.error(
                        "Some S2 band you want to remove are not valid. "
                        "Please check that all of them are valid or leave the list empty "
                        "if you do not want to remove any bands."
                    )
                else:
                    num_channels -= len(self.remove_s2_bands)

        self.total_num_channels = num_channels

        if self.normalize is False:
            logger.warning(
                "You have deactivated the default normalization. "
                "This requires custom normalization and modification of"
                " sequence padding with a correct padding value (default 0.0)."
            )
