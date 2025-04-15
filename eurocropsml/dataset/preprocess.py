"""Preprocessing utilities for the EuroCrops dataset."""

import calendar
import logging
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from eurocropsml.acquisition.config import S1_BANDS, S2_BANDS
from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.dataset.download import download_dataset

logger = logging.getLogger(__name__)


def _read_geojson_file(metadata_file_path: Path, country: str) -> gpd.GeoDataFrame:
    df = gpd.read_file(metadata_file_path.joinpath(f"{country}.geojson")).set_index("parcel_id")
    # get centroid in projected (flat) coordinates, then convert to spherical (lat, lon)
    df["geometry"] = df["geometry"].to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
    return df


def _read_parquet_file(metadata_file_path: Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_parquet(metadata_file_path)[
        ["parcel_id", "EC_hcat_c", "EC_hcat_n"]
    ].set_index("parcel_id")

    return df


def _get_labels(
    metadata_dir: Path,
    country: str,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> dict[int, int]:
    metadata_df = read_metadata(
        metadata_dir,
        country,
    )
    # Preprocess Classes
    # Only keep classes in `keep_classes`
    if keep_classes := preprocess_config.keep_classes:
        metadata_df = metadata_df[metadata_df.EC_hcat_c.isin(keep_classes)]
    elif excl_classes := preprocess_config.excl_classes:
        metadata_df = metadata_df[~metadata_df.EC_hcat_c.isin(excl_classes)]

    # save series to dictionary with parcel IDs (index) as keys
    parcel_labels: dict[int, int] = metadata_df["EC_hcat_c"].to_dict()

    return parcel_labels


def read_metadata(
    metadata_dir: Path, country: str | None = None, num_workers: int | None = None
) -> pd.DataFrame:
    """Read all metadata from pickle files into a Dataframe.

    Args:
        metadata_dir: Directory containing parcel metadata.
        country: Country to load labels for.
        num_workers: Number of workers used for multiprocessing.

    Returns:
        DataFrame with parcel and corresponding labels.
    """
    if not metadata_dir.is_dir():
        logger.error(f"Directory for classes {metadata_dir} not found!")
        sys.exit(1)

    # Combine all available geojson files
    if country is None:
        with Pool(processes=num_workers) as p:
            dfs = p.map(_read_parquet_file, metadata_dir.rglob("*.parquet"))

        df_classes: pd.DataFrame = pd.concat(dfs, axis=0)

    else:
        df_classes = _read_parquet_file(metadata_dir.joinpath(f"{country}_labels.parquet"))

    df_classes = df_classes.dropna(subset=["EC_hcat_c"])

    return df_classes


def _get_lonlats(metadata_dir: Path, country: str) -> dict[int, np.ndarray]:
    geometries = _read_geojson_file(metadata_dir, country)
    parcel_lonlats: dict[int, np.ndarray] = {
        k: np.concatenate(v.xy) for k, v in geometries["geometry"].to_dict().items()
    }
    return parcel_lonlats


@cache
def get_class_ids_to_names(raw_data_dir: Path) -> dict[str, str]:
    """Get a dictionary mapping between class identifiers and readable names."""
    labels_df: pd.DataFrame = read_metadata(raw_data_dir)
    unique_labels_df = labels_df.drop_duplicates()
    ids_to_names_dict = unique_labels_df.set_index("EC_hcat_c").to_dict()["EC_hcat_n"]
    return {str(k): v for k, v in ids_to_names_dict.items()}


def _find_padding(array: np.ndarray) -> bool:
    return not np.array_equal(array, np.array([-999] * len(array)))


def _filter_padding(data: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padding_mask = np.apply_along_axis(_find_padding, 1, data)
    return data[padding_mask], dates[padding_mask]


def find_clouds(
    array: np.ndarray,
    band4_idx: int,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> float:
    """Get cloud probabilities multi-spectral timeseries.

    Args:
        array: Array containing multi-spectral data (S2)
        band4_idx: Index of band for in array
        preprocess_config: Preprocess config to receive thresholds

    Returns:
        The probability of clouds in the timeseries.
    """
    band4_t1 = preprocess_config.band4_t1
    band4_t2 = preprocess_config.band4_t2
    if array[band4_idx] / 10000 < band4_t1:
        prob = 0.0
    elif array[band4_idx] / 10000 > band4_t2:
        prob = 1.0
    else:
        prob = round((array[band4_idx] / 10000 - band4_t1) / (band4_t2 - band4_t1), 2)
    return prob


def _filter_clouds(
    data: np.ndarray,
    dates: np.ndarray,
    preprocess_config: EuroCropsDatasetPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    band4_prob_threshold = preprocess_config.band4_prob_threshold
    cloud_mask = (
        np.apply_along_axis(
            partial(find_clouds, band4_idx=3, preprocess_config=preprocess_config),
            1,
            data,
        )
        <= band4_prob_threshold
    )
    return data[cloud_mask], dates[cloud_mask]


def _merge_npz_files(file_name: str, file_paths: list[str], output_dir: Path) -> None:
    combined_data = []
    combined_dates = []
    combined_center = []

    for file_path in file_paths:
        with np.load(file_path) as npz:
            combined_data.append(npz["data"])
            combined_dates.append(npz["dates"])
            combined_center.append(npz["center"])

    merged_data = np.concatenate(combined_data, axis=0)
    merged_dates = np.concatenate(combined_dates, axis=0)
    merged_center = np.concatenate(combined_center, axis=0)

    # Sort by dates and ensure unique dates
    sorted_indices = np.argsort(merged_dates)
    merged_dates = merged_dates[sorted_indices]
    merged_data = merged_data[sorted_indices]

    unique_indices = np.unique(merged_dates, return_index=True)[1]
    merged_dates = merged_dates[unique_indices]
    merged_data = merged_data[unique_indices]

    output_path = output_dir.joinpath(file_name)

    np.savez(output_path, data=merged_data, dates=merged_dates, center=merged_center)


def _save_row(
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    preprocess_dir: Path,
    labels: dict[int, int],
    points: dict[int, np.ndarray],
    region: str,
    num_bands: int,
    row_data: tuple[int, pd.Series],
) -> None:
    parcel_id, parcel_data_series = row_data
    timestamps, observations = zip(*parcel_data_series.items())

    if not np.all(observations == np.array([-999] * num_bands)):
        data = np.stack(observations)
        dates = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[D]")
        data, dates = _filter_padding(data, dates)

        if preprocess_config.satellite == "S2" and preprocess_config.filter_clouds:
            data, dates = _filter_clouds(data, dates, preprocess_config)
        if not np.all(data == np.array([-999] * num_bands)):
            label = labels[parcel_id]
            center = points[parcel_id]
            file_dir = preprocess_dir / f"{region}_{str(parcel_id)}_{str(label)}.npz"

            np.savez(file_dir, data=data, dates=dates, center=center)


def preprocess(
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    nuts_level: Literal[1, 2, 3] = 3,
) -> None:
    """Run preprocessing."""

    num_workers: int | None = preprocess_config.num_workers
    satellite: str = preprocess_config.satellite
    raw_data_dir: Path = preprocess_config.raw_data_dir
    final_raw_data_dir: Path = (
        preprocess_config.raw_data_dir / satellite / str(preprocess_config.year)
    )
    preprocess_dir: Path = (
        preprocess_config.preprocess_dir / satellite / str(preprocess_config.year)
    )

    if satellite == "S1":
        logger.info(
            "The current release does not support the collection of Sentinel-1 data. If "
            "you wish to pre-process your own Sentinel-1 data, please be aware that missing"
            " observations should show 'None' such that they will be padded correctly "
            "during pre-processing."
        )
    if preprocess_config.bands is None:
        bands = S2_BANDS if satellite == "S2" else S1_BANDS
    else:
        bands = preprocess_config.bands

    if final_raw_data_dir.exists():
        logger.info("Raw data directory exists. Skipping download.")

        logger.info(f"Starting pre-processing of {satellite} data for {preprocess_config.year}...")
        preprocess_dir.mkdir(exist_ok=True, parents=True)

        month_dir_list = list(final_raw_data_dir.iterdir())
        month_dir_list.sort()

        for month_data_dir in tqdm(month_dir_list, total=len(month_dir_list)):
            if month_data_dir.stem != "allyear":
                month_name: str = calendar.month_name[int(month_data_dir.stem)]
                logger.info(f"Processing data for {month_name}:")
            else:
                month_name = "allyear"
                logger.info("Processing data for full year:")
            month_preprocess_dir: Path = preprocess_dir.joinpath(month_data_dir.stem)
            month_preprocess_dir.mkdir(exist_ok=True, parents=True)

            for file_path in month_data_dir.glob("*.parquet"):
                country_file: pd.DataFrame = pd.read_parquet(file_path).set_index("parcel_id")
                cols = country_file.columns.tolist()
                cols = cols[5:]
                # filter nan-values
                country_file = country_file[~country_file[f"nuts{nuts_level}"].isna()]
                points = _get_lonlats(
                    raw_data_dir.joinpath("geometries", str(preprocess_config.year)), file_path.stem
                )
                labels = _get_labels(
                    raw_data_dir.joinpath("labels", str(preprocess_config.year)),
                    file_path.stem,
                    preprocess_config,
                )

                regions = country_file[f"nuts{nuts_level}"].unique()
                te = tqdm(
                    total=len(regions),
                    desc=f"Processing {file_path.stem}",
                )
                for region in regions:
                    if any(
                        f.name.startswith(region)
                        for f in month_preprocess_dir.iterdir()
                        if f.is_file()
                    ):
                        logger.info(
                            f"There is already existing data for NUTS region {region} for "
                            f"{month_name}. Skipping pre-processing."
                        )
                        continue
                    region_data = country_file[country_file[f"nuts{nuts_level}"] == region]

                    # remove parcels that do not appear in the labels dictionary as keys
                    region_data = region_data[region_data.index.isin(labels.keys())]
                    region_data = region_data[cols]
                    # removing empty columns
                    region_data = region_data.dropna(axis=1, how="all")
                    # removing empty parcels
                    region_data = region_data.dropna(how="all")
                    # replacing single empty timesteps

                    region_data = region_data.apply(
                        lambda x, b=len(bands): x.map(
                            lambda y: np.array([-999] * b) if y is None else y
                        )
                    )
                    with Pool(processes=num_workers) as p:
                        func = partial(
                            _save_row,
                            preprocess_config,
                            month_preprocess_dir,
                            labels,
                            points,
                            region,
                            len(bands),
                        )
                        process_iter = p.imap(func, region_data.iterrows(), chunksize=1000)
                        ti = tqdm(total=len(region_data), desc=f"Processing {region}")
                        _ = [ti.update(n=1) for _ in process_iter]
                        ti.close()

                        te.update(n=1)
                te.close()

        monthly_groups = defaultdict(list)
        for folder in tqdm(preprocess_dir.iterdir(), desc="Merging time series..."):
            if folder.is_dir():
                for npz_file in folder.glob("*.npz"):
                    monthly_groups[npz_file.name].append(npz_file)

        te = tqdm(total=len(monthly_groups), desc="Merging time series...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_merge_npz_files, file_name, file_paths, preprocess_dir)
                for file_name, file_paths in monthly_groups.items()
            ]

            for _ in futures:
                te.update(n=1)

        for folder in preprocess_dir.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder)

        logger.info(f"Data has been preprocessed and saved under {preprocess_dir}.")
    else:
        if preprocess_config.year in [2021]:
            download = typer.confirm(
                "Could not find raw data to preprocess. Would you like to download it?"
            )
        else:
            download = typer.confirm(
                "Could not find raw data to preprocess and Zenodo only has "
                "data for 2021. Do you want to download data for 2021? Otherwise you need "
                "to collect the raw data yourself using the acquisition pipeline."
            )

        if download:
            logger.info("Downloading dataset...")
            download_dataset(preprocess_config)
            logger.info(
                f"Data has been downloaded and saved under {preprocess_config.raw_data_dir.parent}."
            )
        else:
            logger.info(
                "Cannot preprocess without raw data. If you have your own raw data, "
                f"please move it into {preprocess_config.raw_data_dir.parent} and "
                "restart the preprocessing afterwards."
            )
            sys.exit(1)
