"""Building the EuroCropsML dataset."""

import logging
from pathlib import Path
from typing import cast

from eurocropsml.acquisition import clipper, collector, copier, region
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
    final_output_dir = config.raw_data_dir
    output_dir = config.output_dir
    local_dir = config.local_dir

    vector_data_dir = Settings().data_dir.joinpath(
        "meta_data",
        "vector_data",
    )

    config.country_config.post_init(vector_data_dir)
    ct_config = config.country_config

    country = ct_config.country

    shape_dir_clean: Path = vector_data_dir.joinpath(
        f"{cast(Path, ct_config.shapefile_folder).name}_clean",
    )

    shape_dir_clean.mkdir(exist_ok=True, parents=True)
    shape_dir_clean = shape_dir_clean.joinpath(f"{cast(Path, ct_config.shapefile).name}")

    nuts_dir = Settings().data_dir.joinpath("meta_data", "NUTS")

    country_output_dir: Path = output_dir.joinpath(country.replace(" ", "_"))
    country_output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Processing year {ct_config.year} for {country}.")

    collector.acquire_s2_tiles(
        ct_config,
        country_output_dir.joinpath("collector"),
        cast(Path, ct_config.shapefile),
        shape_dir_clean,
        config.eodata_dir,
        config.workers,
    )
    logger.info("Finished step 1: Acquiring list of necessary SAFE-files.")
    copier.merge_s2_safe_files(
        cast(list[str], ct_config.bands), country_output_dir, config.workers, local_dir
    )
    if local_dir is not None:
        logger.info(
            "Finished step 2: Copying SAFE-files to local disk and "
            "acquiring list of individual band image paths."
        )
    else:
        logger.info("Finished step 2: Acquiring list of individual band image paths.")

    clipper.clipping(
        ct_config,
        country_output_dir,
        shape_dir_clean,
        config.workers,
        config.chunk_size,
        config.multiplier,
        local_dir,
    )

    logger.info("Finished step 3: Clipping parcels from raster tiles.")
    region.add_nuts_regions(
        ct_config,
        country_output_dir,
        shape_dir_clean,
        nuts_dir,
        final_output_dir,
    )
    logger.info("Finished step 4: Adding NUTS regions to final DataFrame.")
