"""Preprocessing utilities for the EuroCrops dataset."""

import logging
import sys
from functools import cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import typer
from tqdm import tqdm

from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.utils import _unzip_file

logger = logging.getLogger(__name__)


def download_dataset(preprocess_config: EuroCropsDatasetPreprocessConfig) -> None:
    """Download EuroCropsML dataset from Zenodo.

    Args:
        preprocess_config: Config used to access the Zenodo URL.

    Raises:
        requests.exceptions.HTTPError: If Zenodo records could not be accessed.
    """

    response = requests.get(preprocess_config.download_url)

    data_dir: Path = preprocess_config.raw_data_dir.parent
    data_dir.mkdir(exist_ok=True, parents=True)

    try:
        response.raise_for_status()
        data = response.json()

        for file_entry in data["files"]:
            file_url = file_entry["links"]["self"]
            file_name = file_entry["key"]

            filepath: Path = data_dir.joinpath(file_name)

            if not filepath.exists():

                response = requests.get(file_url)
                response.raise_for_status()
                with open(filepath, "wb") as file:
                    file.write(response.content)
                logger.info(f"{filepath.name} downloaded.")
            else:
                logger.info(f"{filepath.name} already exists. Skipping downloading.")
            file_dir: Path = filepath.with_suffix("")
            if not file_dir.exists():
                file_dir = filepath.parent
                _unzip_file(data_dir.joinpath(file_name), file_dir)
                logger.info(f"{filepath.name} unzipped.")
            else:
                logger.info(f"{file_dir} already exists. Skipping unzipping.")
    except requests.exceptions.HTTPError as err:
        logger.info(f"There was an error when trying to get the Zenodo record: {err}")


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


def _get_latlons(metadata_dir: Path, country: str) -> dict[int, np.ndarray]:
    geometries = _read_geojson_file(metadata_dir, country)
    parcel_latlons: dict[int, np.ndarray] = {
        k: np.concatenate(v.xy) for k, v in geometries["geometry"].to_dict().items()
    }
    return parcel_latlons


@cache
def get_class_ids_to_names(raw_data_dir: Path) -> dict[str, str]:
    """Get a dictionary mapping between class identifiers and readable names."""
    labels_df: pd.DataFrame = read_metadata(raw_data_dir)
    unique_labels_df = labels_df.drop_duplicates()
    ids_to_names_dict = unique_labels_df.set_index("EC_hcat_c").to_dict()["EC_hcat_n"]
    return {str(k): v for k, v in ids_to_names_dict.items()}


def _find_padding(array: np.ndarray) -> bool:
    if np.array_equal(array, np.array([0] * len(array))):
        return False
    return True


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


def _save_row(
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    preprocess_dir: Path,
    labels: dict[int, int],
    points: dict[int, np.ndarray],
    region: str,
    row_data: tuple[int, pd.Series],
) -> None:
    parcel_id, parcel_data_series = row_data
    timestamps, observations = zip(*parcel_data_series.items())
    if not np.all(observations == np.array([0] * 13)):
        data = np.stack(observations)
        dates = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[D]")
        data, dates = _filter_padding(data, dates)

        if preprocess_config.filter_clouds:
            data, dates = _filter_clouds(data, dates, preprocess_config)
        if not np.all(data == np.array([0] * 13)):
            label = labels[parcel_id]
            center = points[parcel_id]
            file_dir = preprocess_dir / f"{region}_{str(parcel_id)}_{str(label)}.npz"

            np.savez(file_dir, data=data, dates=dates, center=center)


def preprocess(
    preprocess_config: EuroCropsDatasetPreprocessConfig,
    nuts_level: Literal[1, 2, 3] = 3,
) -> None:
    """Run preprocessing."""

    raw_data_dir = preprocess_config.raw_data_dir
    preprocess_dir = preprocess_config.preprocess_dir
    num_workers = preprocess_config.num_workers

    if preprocess_dir.exists() and len(list((preprocess_dir.iterdir()))) > 0:
        logger.info("Preprocessing directory already exists. Nothing to do.")
        sys.exit(0)

    if raw_data_dir.exists():
        logger.info("Download directory exists. Skipping download.")

        logger.info("Starting preprocessing. Compiling labels and centerpoints of parcels")
        preprocess_dir.mkdir(exist_ok=True, parents=True)
        for file_path in raw_data_dir.glob("*.parquet"):
            country_file: pd.DataFrame = pd.read_parquet(file_path).set_index("parcel_id")
            cols = country_file.columns.tolist()
            cols = cols[5:]
            # filter nan-values
            country_file = country_file[~country_file[f"nuts{nuts_level}"].isna()]
            points = _get_latlons(raw_data_dir.joinpath("geometries"), file_path.stem)
            labels = _get_labels(raw_data_dir.joinpath("labels"), file_path.stem, preprocess_config)

            # country_file.set_index("parcel_id", inplace=True)
            regions = country_file[f"nuts{nuts_level}"].unique()
            te = tqdm(
                total=len(regions),
                desc=f"Processing {file_path.stem}",
            )
            for region in regions:
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
                    lambda x: x.map(lambda y: np.array([0] * 13) if y is None else y)
                )
                with Pool(processes=num_workers) as p:
                    func = partial(
                        _save_row,
                        preprocess_config,
                        preprocess_dir,
                        labels,
                        points,
                        region,
                    )
                    process_iter = p.imap(func, region_data.iterrows(), chunksize=1000)
                    ti = tqdm(total=len(region_data), desc=f"Processing {region}")
                    _ = [ti.update(n=1) for _ in process_iter]
                    ti.close()

                    te.update(n=1)
            te.close()

        logger.info(f"Data has been preprocessed and saved under {preprocess_dir}.")
    else:
        download = typer.confirm(
            "Could not find raw data to preprocess. "
            "Would you like to download it? This will also download a preprocessed version."
        )
        if download:
            logger.info("Downloading dataset.")
            download_dataset(preprocess_config)
            logger.info(
                f"Data has been downloaded and saved under {preprocess_config.raw_data_dir.parent}."
            )
        else:
            logger.info("Cannot preprocess without raw data.")
            sys.exit(1)
