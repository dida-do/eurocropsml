"""Utilities for clipping polygons from raster tiles."""

import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Literal, cast

import esa_snappy as snappy
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from esa_snappy import Product, ProductIO
from pyproj import CRS
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

from eurocropsml.acquisition.clipping.s1_preprocessing import (
    do_apply_orbit_file,
    do_calibration,
    do_speckle_filtering,
    do_subset,
    do_terrain_correction,
    do_thermal_noise_removal,
)
from eurocropsml.acquisition.config import CollectorConfig

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


def _transform_polygon(
    polygon: Polygon | MultiPolygon, inv_transform: Affine
) -> Polygon | MultiPolygon:
    if isinstance(polygon, MultiPolygon):
        # transform MultiPolygons Polygon by Polygon
        transformed_polygons = []
        for single_polygon in polygon.geoms:
            # transformation of exterior coordinates
            transformed_exterior = [
                inv_transform * (x, y) for x, y in single_polygon.exterior.coords
            ]
            # transformation of interior coordinates
            transformed_interiors = [
                [inv_transform * (x, y) for x, y in interior.coords]
                for interior in single_polygon.interiors
            ]

            transformed_polygons.append(Polygon(transformed_exterior, transformed_interiors))

        return MultiPolygon(transformed_polygons)

    else:
        # transform Polygons
        transformed_exterior = [inv_transform * (x, y) for x, y in polygon.exterior.coords]
        transformed_interiors = [
            [inv_transform * (x, y) for x, y in interior.coords] for interior in polygon.interiors
        ]

        return Polygon(transformed_exterior, transformed_interiors)


def mask_polygon_raster(
    satellite: Literal["S1", "S2"],
    tilepaths: list[Path],
    polygon_df: pd.DataFrame,
    parcel_id_name: str,
    product_date: str,
) -> pd.DataFrame:
    """Clipping parcels from raster files (per band) and calculating median pixel value per band.

    Args:
        satellite: S1 for Sentinel-1 and S2 for Sentinel-2.
        tilepaths: Paths to the raster's band tiles.
        config: CollectorConfig used for pre-processing Sentinel-1
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
        if satellite == "S2":
            try:
                with rasterio.open(band_path, "r") as raster_tile:
                    if b == 0:
                        if polygon_df.crs.srs != raster_tile.crs.data["init"]:
                            # transforming shapefile into CRS of raster tile
                            polygon_df = polygon_df.to_crs(raster_tile.crs.data["init"])

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{band_path} could not be found. Please make sure it is "
                    "present, delete all clipped files and rerun the whole "
                    "clipping process."
                )

            polygon_df.apply(
                lambda row: _process_row(row, raster_tile, parcels_dict, parcel_id_name, satellite),
                axis=1,
            )

        else:
            safe_file = Path(band_path).parent.parent

            manifest_path = str(safe_file) + "/manifest.safe"
            sentinel_1 = ProductIO.readProduct(str(manifest_path))

            wkt = sentinel_1.getSceneCRS().toWKT()
            authority_start = wkt.find('AUTHORITY["EPSG",')
            epsg_start = wkt.find('"', authority_start) + 1
            epsg_end = wkt.find("]", epsg_start)
            epsg_code = wkt[epsg_start:epsg_end].split(",")[1].strip('"')
            crs = f"EPSG:{epsg_code}"

            polarization = "DV"
            pols = "VV,VH"

            orbit_applied = do_apply_orbit_file(sentinel_1)
            thermal_removed = do_thermal_noise_removal(orbit_applied)
            calibrated = do_calibration(thermal_removed, polarization, pols)
            down_filtered = do_speckle_filtering(calibrated)
            down_tercorrected = do_terrain_correction(down_filtered, 1)

            del orbit_applied, thermal_removed, calibrated

            band = "VV" if "vv" in band_path else "VH"

            if band == "VV":
                if polygon_df.crs.srs != crs:
                    # transforming shapefile into CRS of band product
                    polygon_df = polygon_df.to_crs(crs)

                # TODO: change to esa-snappy getPixelPos
                with rasterio.open(band_path, "r") as raster_tile:

                    gcps, gcps_crs = raster_tile.get_gcps()

                    transform = rasterio.transform.from_gcps(gcps)

                    inv_transform = ~transform  # Invert the affine transformation matrix

                    polygon_df["geometry_new"] = polygon_df["geometry"].apply(
                        lambda poly, i_trans=inv_transform: _transform_polygon(poly, i_trans)
                    )

            polygon_df.apply(
                lambda row: _process_row(
                    row, down_tercorrected, parcels_dict, parcel_id_name, satellite, band
                ),
                axis=1,
            )
            del down_tercorrected

    parcels_df = pd.DataFrame(list(parcels_dict.items()), columns=[parcel_id_name, product_date])

    return parcels_df


def suppress_output(func, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        # Save original file descriptors
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)
        try:
            # Redirect stdout and stderr to /dev/null
            os.dup2(fnull.fileno(), 1)
            os.dup2(fnull.fileno(), 2)
            return func(*args, **kwargs)
        finally:
            # Restore original file descriptors
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)


def _process_row(
    row: pd.Series,
    band_tile: DatasetReader | ProductIO,
    parcels_dict: dict[int, list[float | None]],
    parcel_id_name: str,
    satellite: Literal["S1", "S2"],
    band: str = "",
) -> None:
    """Masking geometry from raster tiles and calculating median pixel value."""
    parcel_id: int = row[parcel_id_name]
    geom = row["geometry_new"]

    try:
        if satellite == "S1":
            cropped_img = suppress_output(do_subset, band_tile, row["geometry"].wkt)
            band_img = cropped_img.getBand(f"Sigma0_{band}")

            raster_width = band_img.getRasterWidth()
            raster_height = band_img.getRasterHeight()
            buffer = np.zeros(raster_width * raster_height, dtype=np.float32)

            band_img.loadRasterData()
            pixels = band_img.getPixels(0, 0, raster_width, raster_height, buffer)
            pixels = np.array(pixels, dtype=np.float32).reshape((raster_height, raster_width))
            not_zero = pixels[:, :] != 0
            del cropped_img

            if not not_zero.any():
                patch_median: float | None = None
            else:
                patch_median = np.median(pixels[not_zero]).astype(np.float32)

            parcels_dict[parcel_id].append(patch_median)

        else:
            masked_img, masked_transform = mask(band_tile, [geom], crop=True, nodata=0)
            masked_img = reshape_as_image(masked_img)
            # third dimension has index 0 since we only have one band
            not_zero = masked_img[:, :, 0] != 0

            if not not_zero.any():
                patch_median = 0.0
            else:
                # If no calibration data is available (Sentinel-2), use masked values directly
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
    full_df.set_index(parcel_id_name, inplace=True)
    for file in tqdm(df_list):
        full_df = full_df.fillna(pd.read_pickle(file))

    # reset index column
    full_df = full_df.reset_index()

    full_df.to_parquet(output_dir.joinpath("clipped.parquet"))
    logger.info("Saved final clipped file.")
