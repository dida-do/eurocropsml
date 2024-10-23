"""Clipping parcel polygons from raster files."""

import concurrent.futures
import gc
import logging
import multiprocessing as mp_orig
import pickle
from functools import partial
from pathlib import Path
from typing import Literal, cast

import geopandas as gpd
import pandas as pd
import pyogrio
import typer
from tqdm import tqdm

from eurocropsml.acquisition.config import CollectorConfig
from eurocropsml.acquisition.utils import _merge_clipper, mask_polygon_raster

logger = logging.getLogger(__name__)


def _merge_dataframe(
    full_df: gpd.GeoDataFrame,
    clipped_output_dir: Path,
    output_dir: Path,
    parcel_id_name: str,
    new_data: bool,
) -> None:
    """Merging all clipped parcels into one final DataFrame.

    full_df: Final dataframe containing all parcels without clipped values.
    clipped_output_dir: Directory where the individual clipped parcels are stored.
    output_dir: Directory to save the final output.
    parcel_id_name: Country-specific parcel ID name.
    new_data: Whether new data has been processed.

    """
    if not output_dir.joinpath("clipped.parquet").exists():
        _merge_clipper(
            full_df,
            clipped_output_dir,
            output_dir,
            parcel_id_name,
        )
    else:
        if new_data:
            reprocess = typer.confirm(
                f"{output_dir.joinpath('clipped.parquet')} already exists but new data has been "
                "processed. Do you want to reprocess the file?"
            )
        else:
            reprocess = typer.confirm(
                f"{output_dir.joinpath('clipped.parquet')} already exists and no new data has been"
                " processed. Do you want to reprocess the file?"
            )

        if reprocess:
            logger.info("File is being deleted and reprocessed.")
            output_dir.joinpath("clipped.parquet").unlink()
            _merge_clipper(
                full_df,
                clipped_output_dir,
                output_dir,
                parcel_id_name,
            )


def _get_arguments(
    config: CollectorConfig,
    workers: int,
    shape_dir: Path,
    output_dir: Path,
    month: int,
    local_dir: Path | None = None,
) -> tuple[list[tuple[pd.DataFrame, list]], gpd.GeoDataFrame, Path]:
    """Get arguments for clipping polygons from raster files.

    Args:
        config: Country-specific configuration for acquiring EuroCrops reflectance data.
        workers: Maximum number of workers used for multiprocessing.
        shape_dir: Directory where EuroCrops shapefile is stored.
        output_dir: Directory to get the list of .SAFE files from and to store the
            argument list.
        local_dir: Local directory where the .SAFE files were copied to.

    Returns:
        - List of tuples of arguments for clipping raster tiles.
        - Geo-Dataframe that contains all parcels and additional information.
        - Output directory

    """

    parcel_id_name: str = cast(str, config.parcel_id_name)
    bands: list[str] = cast(list[str], config.bands)

    clipping_path = output_dir.joinpath("clipper").joinpath(f'{month}')
    clipping_path.mkdir(exist_ok=True, parents=True)

    if clipping_path.joinpath("args.pkg").exists():
        logger.info("Loading argument list for parallel raster clipping.")
        with open(clipping_path.joinpath("args.pkg"), "rb") as file:
            args: list[tuple[pd.DataFrame, list]] = pickle.load(file)
        shapefile: gpd.GeoDataFrame = pd.read_pickle(clipping_path.joinpath("empty_polygon_df.pkg"))
    else:
        logger.info("No argument list found. Will create it.")
        # DataFrame of raster file/parcel matches
        full_images_paths: Path = output_dir.joinpath("collector", "full_parcel_list.pkg")
        full_images = pd.read_pickle(full_images_paths)

        full_images['completionDate'] = pd.to_datetime(full_images['completionDate'])
        full_images = full_images[(full_images['completionDate'].dt.month == month)]

        if local_dir is not None:
            full_images["productIdentifier"] = str(local_dir) + full_images[
                "productIdentifier"
            ].astype(str)

        band_image_path: Path = output_dir.joinpath("copier", "band_images.pkg")
        band_images: pd.DataFrame = pd.read_pickle(band_image_path)

        band_images = band_images[
            (band_images['productIdentifier'].str.extract(r'/\d{4}/(\d{2})/')[0].astype(int) == month)
        ]

        max_workers = min(mp_orig.cpu_count(), max(1, min(len(band_images), workers)))
        args = []
        with mp_orig.Pool(processes=max_workers) as p:
            func = partial(_filter_args, bands, full_images)
            process_iter = p.imap(func, band_images.iterrows())
            ti = tqdm(
                total=len(band_images),
                desc="Building argument list for parallel raster clipping.",
            )
            for result in process_iter:
                args.append(result)
                ti.update(n=1)
            ti.close()

        with open(clipping_path.joinpath("args.pkg"), "wb") as fp:
            pickle.dump(args, fp)
        logger.info("Saved argument list.")

        date_list = list(full_images["completionDate"].unique())
        cols = [parcel_id_name, "geometry"] + date_list

        shapefile = pyogrio.read_dataframe(shape_dir)

        shapefile[[parcel_id_name, "geometry"]]

        shapefile = shapefile.reindex(columns=cols)

        shapefile.to_pickle(clipping_path.joinpath("empty_polygon_df.pkg"))

    shapefile[parcel_id_name] = shapefile[parcel_id_name].astype(int)

    return args, shapefile, clipping_path


def _filter_args(
    bands: list[str], full_images: pd.DataFrame, band_image: tuple[int, pd.Series]
) -> tuple[pd.DataFrame, list]:
    band_image_row = band_image[1]
    filtered_images = full_images[
        full_images["productIdentifier"] == band_image_row["productIdentifier"]
    ]
    band_tiles = [band_image_row[f"bandImage_{band}"] for band in bands]
    return filtered_images, band_tiles


def _process_raster_parallel(
    satellite: Literal["S1", "S2"],
    polygon_df: pd.DataFrame,
    parcel_id_name: str,
    filtered_images: gpd.GeoDataFrame,
    band_tiles: list[Path],
) -> pd.DataFrame:
    """Processing one raster file.

    Args:
        satellite: S1 for Sentinel-1 and S2 for Sentinel-2.
        polygon_df: Dataframe containing all parcel ids. Will be merged with the clipped values.
        parcel_id_name: The country's parcel ID name (varies from country to country).
        filtered_images: Dataframe containing all parcel ids that lie in this raster tile.
        band_tiles: Paths to the raster's band tiles.

    Returns:
        Final DataFrame with clipped parcel reflectance values for this given raster tile.
    """

    try:
        # all parcel ids that match product Identifier
        parcel_ids = list(filtered_images[parcel_id_name])
        parcel_ids = [int(id) for id in parcel_ids]
        # observation date
        product_date = str(filtered_images["completionDate"].unique()[0])

        # geometry information of all parcels
        filtered_geom = polygon_df[polygon_df[parcel_id_name].isin(parcel_ids)]

        result = mask_polygon_raster(
            satellite, band_tiles, filtered_geom, parcel_id_name, product_date
        )

        if result is not None:
            result.set_index(parcel_id_name, inplace=True)
        else:
            result = pd.DataFrame()

    except Exception:
        result = pd.DataFrame()

    return result


def clipping(
    config: CollectorConfig,
    output_dir: Path,
    shape_dir: Path,
    workers: int,
    chunk_size: int,
    multiplier: int,
    local_dir: Path | None = None,
) -> None:
    """Main function to conduct polygon clipping.

    Args:
        config: Country-specific configuration for acquiring EuroCrops reflectance data.
        output_dir: Directory path where intermediate results will be stored.
        shape_dir: Directory path where EuroCrops shapefile is stored.
        workers: Maximum number of workers used for multiprocessing.
        chunk_size: Chunk size used for multiprocessed raster clipping.
        multiplier: Intermediate results will be saved every multiplier steps.
        local_dir: Local directory where the .SAFE files were copied to.
    """

    for month in config.months:

        args_month, polygon_df, clipping_path = _get_arguments(
            config=config,
            workers=workers,
            shape_dir=shape_dir,
            output_dir=output_dir,
            month=month,
            local_dir=local_dir,
        )

        clipped_dir = clipping_path.joinpath("clipped")
        clipped_dir.mkdir(exist_ok=True, parents=True)

        max_workers = min(mp_orig.cpu_count(), max(1, min(len(args_month), workers)))

        # Process data in smaller chunks
        file_counts = len(list(clipped_dir.rglob("Final_*.pkg")))

        processed = file_counts * multiplier * chunk_size
        save_files = multiplier * chunk_size
        file_counts += 1

        polygon_df[config.parcel_id_name] = polygon_df[config.parcel_id_name].astype(int)

        func = partial(
            _process_raster_parallel, config.satellite, polygon_df, cast(str, config.parcel_id_name)
        )

        polygon_df_month = polygon_df.drop(["geometry"], axis=1)
        df_final_month = polygon_df_month.copy()
        df_final_month.set_index(config.parcel_id_name, inplace=True)

        polygon_df = polygon_df.drop(["geometry"], axis=1)

        new_data: bool = False
        if processed < len(args_month):
            new_data = True
            logger.info(f"Starting parallel raster clipping for {month}...")
            te = tqdm(total=len(args_month) - processed, desc=f"Clipping raster tiles for {month}.")
            while processed < len(args_month):
                chunk_args: list[tuple[pd.DataFrame, list]] = args_month[processed : processed + chunk_size]
                results: list[pd.DataFrame] = []

                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(func, *arg) for arg in chunk_args]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception:
                            results.append(None)

                # Process and save results
                for result in results:
                    if result is not None and not result.empty:
                        result.columns = [pd.to_datetime(result.columns[0]).strftime('%Y-%m-%d')]
                        df_final_month.columns = [pd.to_datetime(col).strftime('%Y-%m-%d') for col in
                                                  df_final_month.columns]
                        df_final_month = df_final_month.fillna(result)
                    te.update(n=1)

                processed += len(chunk_args)
                if processed % save_files == 0:
                    df_final_month.to_pickle(clipped_dir.joinpath(f"Final_{file_counts}.pkg"))
                    del df_final_month
                    df_final_month = polygon_df_month.copy()
                    df_final_month.set_index(config.parcel_id_name, inplace=True)
                    file_counts += 1
                # Clear variables to release memory
                del chunk_args, futures
                gc.collect()

            df_final_month.to_pickle(clipped_dir.joinpath(f"Final_{file_counts}.pkg"))

        _merge_dataframe(
            polygon_df,
            clipped_dir,
            clipped_dir,
            cast(str, config.parcel_id_name),
            new_data,
        )

def main():
    config = CollectorConfig(country="Austria", year=2021)
    vector_data_dir = Path('/big_volume/data_s1/meta_data/vector_data/')
    config.post_init(vector_data_dir)
    output_dir = Path('/big_volume/data_s1/output_data/Austria/S1')
    shape_dir = Path('/big_volume/data_s1/meta_data/vector_data/AT_2021_clean/')
    workers = 16
    chunk_size = 20
    multiplier = 15

    clipping(config, output_dir, shape_dir, workers, chunk_size, multiplier)

if __name__ == '__main__':
    main()
