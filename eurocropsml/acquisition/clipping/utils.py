"""Utilities for clipping polygons from raster tiles."""

import logging
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from rasterio.features import geometry_window
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

from eurocropsml.acquisition.clipping.calibration import (
    _calibrate_digital_number_in_db,
    _open_calibration_dataset,
    _open_noise_dataset,
)

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
    bands: list[str],
    polygon_df: pd.DataFrame,
    parcel_id_name: str,
    product_date: str,
    denoise: bool = True,
) -> pd.DataFrame:
    """Clipping parcels from raster files (per band) and calculating median pixel value per band.

    Args:
        satellite: S1 for Sentinel-1 and S2 for Sentinel-2.
        tilepaths: Paths to the raster's band tiles.
        bands: (Sub-)set of Sentinel-1 (radar) or Sentinel-2 (spectral) bands.
        polygon_df: GeoDataFrame of all parcels to be clipped.
        parcel_id_name: The country's parcel ID name (varies from country to country).
        product_date: Date on which the raster tile was obtained.
        denoise: Whether to perform thermal noise removal for Sentinel-1.

    Returns:
        Dataframe with clipped values.
    """

    parcels_dict: dict[int, list[float | None]] = {
        parcel_id: [] for parcel_id in polygon_df[parcel_id_name].unique()
    }

    # removing any self-intersections or inconsistencies in geometries
    polygon_df["geometry"] = polygon_df["geometry"].buffer(0)
    polygon_df = polygon_df.reset_index(drop=True)

    sigma_nought: xr.DataArray | None = None
    noise_vector: xr.DataArray | None = None

    for b, band_path in enumerate(tilepaths):
        with rasterio.open(band_path, "r") as raster_tile:
            if b == 0:
                if satellite == "S2" and polygon_df.crs.srs != raster_tile.crs.data["init"]:
                    # transforming shapefile into CRS of raster tile
                    polygon_df = polygon_df.to_crs(raster_tile.crs.data["init"])
                elif satellite == "S1":
                    gcps, _ = raster_tile.get_gcps()
                    transform = rasterio.transform.from_gcps(gcps)

                    inv_transform = ~transform  # Invert the affine transformation matrix

                    polygon_df["geometry"] = polygon_df["geometry"].apply(
                        lambda poly, inv_transform=inv_transform: _transform_polygon(
                            poly, inv_transform
                        )
                    )

                    calibration_data = _open_calibration_dataset(
                        Path(band_path).parent.parent, bands[b]
                    )
                    sigma_nought = calibration_data.data_vars["sigmaNought"]

                    if denoise:
                        noise_data = _open_noise_dataset(Path(band_path).parent.parent, bands[b])
                        noise_vector = noise_data.data_vars["noiseRangeLut"]

            # clippping geometry out of raster tile and saving in dictionary
            polygon_df.apply(
                lambda row, sigma_nought=sigma_nought, noise_vector=noise_vector: _process_row(
                    row, raster_tile, parcels_dict, parcel_id_name, sigma_nought, noise_vector
                ),
                axis=1,
            )

    parcels_df = pd.DataFrame(list(parcels_dict.items()), columns=[parcel_id_name, product_date])

    return parcels_df


def _process_row(
    row: pd.Series,
    raster_tile: DatasetReader,
    parcels_dict: dict[int, list[float | None]],
    parcel_id_name: str,
    sigma_nought: xr.DataArray | None = None,
    noise_vector: xr.DataArray | None = None,
) -> None:
    """Masking geometry from raster tiles and calculating median pixel value."""
    parcel_id: int = row[parcel_id_name]
    geom = row["geometry"]

    try:
        masked_img, _ = mask(raster_tile, [geom], crop=True, nodata=0)
        masked_img = reshape_as_image(masked_img)
        # third dimension has index 0 since we only have one band
        not_zero = masked_img[:, :, 0] != 0

        # Apply calibration to raw digital numbers (DN) in order to receive backscatter in dB
        if sigma_nought is not None:
            if not not_zero.any():
                patch_median: float | None = None
            else:
                band_data = masked_img[:, :, 0]

                # Get the position (row and pixel_col) of the geometry in the original raster
                # 1. create window around geometry in terms of pixel/line coordinates
                window = geometry_window(raster_tile, [geom])

                # 2. Extract start_row, start_col
                start_row, start_col = window.toslices()[0].start, window.toslices()[1].start

                start_row = int(np.floor(start_row))
                start_col = int(np.floor(start_col))

                # 3. Compute the pixel and line coordinates in the original raster's space
                pixel_coords = np.arange(start_col, start_col + band_data.shape[1])
                line_coords = np.arange(start_row, start_row + band_data.shape[0])

                cropped_data = xr.DataArray(
                    data=band_data,
                    dims=("line", "pixel"),
                    coords={"line": line_coords, "pixel": pixel_coords},
                    name="cropped_image",
                )

                # get backscatter in decibels
                backscatter_db: np.ndarray = _calibrate_digital_number_in_db(
                    cropped_data, sigma_nought, noise_vector
                )
                patch_median = np.median(backscatter_db[not_zero])

        else:
            if not not_zero.any():
                patch_median = 0.0
            else:
                # If no calibration data is available (Sentinel-2), use masked values directly
                # Calculate the median of each patch where the clipped values are not zero
                patch_median = np.median(masked_img[not_zero]).astype(np.int16)

                # Ensure the median value is non-negative
                patch_median = max(0, cast(float, patch_median))

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
) -> None:
    logger.info("Starting merging of DataFrames...")
    df_list: list = [file for file in clipped_output_dir.iterdir() if "Final_" in file.name]

    # setting parcel_id column to index
    full_df.set_index(parcel_id_name, inplace=True)
    for file in tqdm(df_list):
        full_df = full_df.fillna(pd.read_pickle(file))

    # reset index column
    full_df = full_df.reset_index()

    full_df.to_parquet(output_dir.joinpath("clipped.parquet"))
    logger.info("Saved final clipped file.")
