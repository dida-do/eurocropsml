"""Utilities for clipping polygons from raster tiles."""

import logging
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from tqdm import tqdm

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


def mask_polygon_raster(
    tilepaths: list[Path],
    polygon_df: pd.DataFrame,
    parcel_id_name: str,
    product_date: str,
) -> pd.DataFrame:
    """Clipping parcels from raster files (per band) and calculating median pixel value per band.

    Args:
        tilepaths: Paths to the raster's band tiles.
        polygon_df: GeoDataFrame of all parcels to be clipped.
        parcel_id_name: The country's parcel ID name (varies from country to country).
        product_date: Date on which the raster tile was obtained.

    Returns:
        Dataframe with clipped values.

    Raises:
        FileNotFoundError: If the raster file cannot be found.
    """

    parcels_dict: dict[int, list[float | None]] = {
        parcel_id: [] for parcel_id in polygon_df[parcel_id_name].unique()
    }

    # removing any self-intersections or inconsistencies in geometries
    polygon_df["geometry"] = polygon_df["geometry"].buffer(0)
    polygon_df = polygon_df.reset_index(drop=True)

    for b, band_path in enumerate(tilepaths):
        with rasterio.open(band_path, "r") as raster_tile:
            if b == 0 and polygon_df.crs.srs != raster_tile.crs:
                # transforming shapefile into CRS of raster tile
                polygon_df = polygon_df.to_crs(raster_tile.crs)

            # clippping geometry out of raster tile and saving in dictionary
            polygon_df.apply(
                lambda row: _process_row(row, raster_tile, parcels_dict, parcel_id_name),
                axis=1,
            )

    parcels_df = pd.DataFrame(list(parcels_dict.items()), columns=[parcel_id_name, product_date])

    return parcels_df


def _process_row(
    row: pd.Series,
    raster_tile: DatasetReader,
    parcels_dict: dict[int, list[float | None]],
    parcel_id_name: str,
) -> None:
    """Masking geometry from raster tiles and calculating median pixel value."""
    parcel_id: int = row[parcel_id_name]
    geom = row["geometry"]

    try:
        masked_img, _ = mask(raster_tile, [geom], crop=True, nodata=0)
        masked_img = reshape_as_image(masked_img)
        # third dimension has index 0 since we only have one band
        not_zero = masked_img[:, :, 0] != 0

        if not not_zero.any():
            patch_median = 0.0
        else:
            # Calculate the median of each patch where the clipped values are not zero
            patch_median = np.median(masked_img[not_zero]).astype(np.float32)

            # Ensure the median value is non-negative
            patch_median = max(0.0, cast(float, patch_median))

        parcels_dict[parcel_id].append(patch_median)

    except ValueError:
        # Since we are cropping to the extent of the geometry, if none of the raster pixels is
        # fully contained inside the geometry, rasterio.mask will throw an error that the shapes
        # do not overlap
        parcels_dict[parcel_id].append(None)


def _merge_clipper(
    full_df: gpd.GeoDataFrame,
    clipped_output_dir: Path,
    output_dir: Path,
    parcel_id_name: str,
    month: str,
) -> None:
    logger.info(f"Starting merging of DataFrames for {month}...")

    df_list: list = [file for file in clipped_output_dir.iterdir() if "Final_" in file.name]

    # setting parcel_id column to index
    if parcel_id_name in full_df.columns:
        full_df.set_index(parcel_id_name, inplace=True)
    for file in tqdm(df_list):
        full_df = full_df.fillna(pd.read_pickle(file))

    # reset index column
    full_df = full_df.reset_index()
    full_df.columns = full_df.columns.astype(str)

    full_df.to_parquet(output_dir.joinpath("clipped.parquet"))
    logger.info("Saved final clipped file.")
