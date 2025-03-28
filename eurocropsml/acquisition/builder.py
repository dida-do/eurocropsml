"""Building the EuroCropsML dataset."""

import logging
from pathlib import Path
from typing import cast

from eurocropsml.acquisition import collector, copier, region
from eurocropsml.acquisition.clipping import clipper
from eurocropsml.acquisition.config import AcquisitionConfig
from eurocropsml.settings import Settings

logger = logging.getLogger(__name__)


def build_dataset(
    config: AcquisitionConfig,
) -> None:
    """Acquiring and preparing EuroCrops dataset for ML purposes.

    Args:
        config: Configuration for acquiring EuroCrops reflectance data.

    """
    vector_data_dir = Settings().data_dir.joinpath(
        "meta_data",
        "vector_data",
    )

    config.country_config.post_init(vector_data_dir)
    ct_config = config.country_config

    final_output_dir = config.raw_data_dir.joinpath(ct_config.satellite, str(ct_config.year))
    output_dir = config.output_dir
    local_dir = config.local_dir

    country = ct_config.country

    shape_dir_clean: Path = vector_data_dir.joinpath(
        f"{cast(Path, ct_config.shapefile_folder).name}_clean",
    )

    shape_dir_clean.mkdir(exist_ok=True, parents=True)
    shape_dir_clean = shape_dir_clean.joinpath(f"{cast(Path, ct_config.shapefile).name}")

    nuts_dir = Settings().data_dir.joinpath("meta_data", "NUTS")

    satellite_output_dir: Path = output_dir.joinpath(country, ct_config.satellite)
    satellite_output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Processing year {ct_config.year} for {country}.")

    collector.acquire_sentinel_tiles(
        ct_config,
        satellite_output_dir.joinpath("collector"),
        cast(Path, ct_config.shapefile),
        shape_dir_clean,
        config.eodata_dir,
        config.workers,
    )
    logger.info("Finished step 1: Acquiring list of necessary .SAFE files.")
    copier.merge_safe_files(
        ct_config.satellite,
        cast(list[str], ct_config.bands),
        satellite_output_dir,
        config.workers,
        local_dir,
    )
    if local_dir is not None:
        logger.info(
            "Finished step 2: Copying .SAFE files to local disk and "
            "acquiring list of individual band image paths."
        )
    else:
        logger.info("Finished step 2: Acquiring list of individual band image paths.")

    clipper.clipping(
        ct_config,
        satellite_output_dir,
        shape_dir_clean,
        config.workers,
        config.chunk_size,
        config.multiplier,
        local_dir,
        config.rebuild,
    )

    logger.info("Finished step 3: Clipping parcels from raster tiles.")
    region.add_nuts_regions(
        ct_config,
        config.raw_data_dir,
        satellite_output_dir,
        shape_dir_clean,
        nuts_dir,
        final_output_dir,
        config.rebuild,
    )
    logger.info("Finished step 4: Adding NUTS regions to final DataFrame.")
