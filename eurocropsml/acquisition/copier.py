"""Copying files to local directories for faster processing."""

import logging
import multiprocessing as mp_orig
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from botocore.client import BaseClient
from tqdm import tqdm

from eurocropsml.acquisition.config import S2_RESOLUTION
from eurocropsml.acquisition.s3 import (
    _download_s3_prefix,
    _establish_s3_client,
    _get_s3_subfolders,
)

logger = logging.getLogger(__name__)


def _copy_to_local_dir(
    source: Literal["eodata", "s3"], local_dir: Path, safe_file: pd.Series
) -> None:
    """Copying files to local directory.

    Args:
        source: Source of the Sentinel tiles. Either directory ('eodata') or S3 bucket ('s3').
        safe_file: File to copy to local directory.

    """
    # Copy all image files from network storage over to local storage
    safe_file_name = safe_file[1]

    if source == "eodata":
        # TODO: check for correctne
        local_product: Path = local_dir.joinpath(safe_file_name.lstrip("/"))
        granule_folder = Path(safe_file_name) / "GRANULE"
        granule_sub_folder = list(granule_folder.iterdir())[0]
        img_data_path = granule_sub_folder / "IMG_DATA"
        local_parent_dir: Path = local_product / "GRANULE" / granule_sub_folder.name / "IMG_DATA"
        if not local_parent_dir.exists():
            local_parent_dir.mkdir(exist_ok=True, parents=True)

        for jp2_file in img_data_path.glob(".jp2"):
            if not jp2_file.exists():
                shutil.copy2(safe_file_name, jp2_file)
    else:
        s3_client: BaseClient = _establish_s3_client()
        granule_prefix = f"{safe_file_name}/GRANULE/"
        granule_subfolders: list = cast(
            list, _get_s3_subfolders(s3_client, granule_prefix, selectionkey="CommonPrefixes")
        )[0]["Prefix"]
        img_data_folder: str = f"{granule_subfolders}IMG_DATA/"
        local_product = local_dir.joinpath(img_data_folder)

        if not local_product.exists():
            local_product.mkdir(parents=True, exist_ok=True)
            _download_s3_prefix(s3_client, img_data_folder, local_product, file_extension=".jp2")


def _get_image_files(
    full_safe_files: pd.DataFrame,
    satellite: Literal["S1", "S2"],
    bands: list[str],
    source: Literal["eodata", "s3"] = "s3",
    local_dir: str = "",
) -> pd.DataFrame:
    """Getting paths for each spectral band.

    Args:
        full_safe_files: DataFrame with .SAFE file paths for which to get the band paths.
        satellite: S1 for Sentinel-1 and S2 for Sentinel-2.
        bands: (Sub-)set of Sentinel-1 (radar) or Sentinel-2 (spectral) bands.
        source: Source of the Sentinel tiles. Either directory ('eodata') or S3 bucket ('s3')
        local_dir: Local directory where the .SAFE files are copied to.
            If None, .SAFE files will not be stored on local disk.

    Returns:
        DataFrame with band paths as columns.

    """

    image_update = pd.DataFrame()

    if source == "eodata":
        for _, row in tqdm(
            full_safe_files.iterrows(),
            total=len(full_safe_files),
            desc="Collecting paths for spectral bands.",
        ):
            if Path(row["productIdentifier"]).exists():
                filename_list: list
                files: list[str]

                if satellite == "S2":
                    filename_list = os.listdir(os.path.join(row["productIdentifier"], "GRANULE"))
                    filename: str = filename_list[0]
                    sub_path: str = os.path.join("GRANULE", filename, "IMG_DATA")
                    path_file: str = os.path.join(row["productIdentifier"], sub_path)

                    files = os.listdir(path_file)

                    for band in bands:
                        if "R10m" in files:
                            res = S2_RESOLUTION[band]
                            r1_files: list[str] = os.listdir(
                                os.path.join(path_file, "R{0}m".format(res))
                            )
                            image_found: list[str] = [
                                file for file in r1_files if f"_B{band}" in file
                            ]
                            if image_found:
                                row["bandImage_{0}".format(band)] = os.path.join(
                                    sub_path, "R{0}m".format(res), image_found[0]
                                )
                                image_found = []

                        else:
                            image_found = [file for file in files if f"_B{band}" in file]
                            row["bandImage_{0}".format(band)] = os.path.join(
                                path_file, image_found[0]
                            )
                            image_found = []

                else:
                    path_file = os.path.join(row["productIdentifier"], "measurement")

                    files = os.listdir(path_file)

                    for i in range(len(bands)):
                        image_found = [file for file in files if f"{bands[i].lower()}" in file]
                        if image_found:
                            row["bandImage_{0}".format(bands[i])] = os.path.join(
                                path_file, image_found[0]
                            )
                        image_found = []

                row_df: pd.DataFrame = row.to_frame().T
                if image_update.empty:
                    image_update = row_df
                else:
                    image_update = pd.concat([image_update, row_df], ignore_index=True)

    else:
        s3_client: BaseClient = _establish_s3_client()
        for _, row in tqdm(
            full_safe_files.iterrows(),
            total=len(full_safe_files),
            desc="Collecting paths for spectral bands from S3 bucket.",
        ):

            prefix = row["productIdentifier"]
            prefix = prefix.replace(local_dir, "")
            granule_prefix = f"{prefix.lstrip('/')}/GRANULE/"

            granule_folder = _get_s3_subfolders(
                s3_client, granule_prefix, selectionkey="CommonPrefixes"
            )
            if not granule_folder:
                logger.info(f"Access to {granule_prefix} failed. Skipping .SAFE file.")
                continue

            sub_folder = cast(list[dict], granule_folder)[0]["Prefix"]
            img_data_prefix = f"{sub_folder}IMG_DATA/"

            img_data_prefixes = _get_s3_subfolders(
                s3_client, img_data_prefix, selectionkey="Contents"
            )
            if not img_data_prefixes:
                logger.info(f"Access to {img_data_prefix} failed. Skipping .SAFE file.")
                continue
            else:
                img_data_prefixes = cast(list[dict], img_data_prefixes)
            # check if 'R10m/' is present in the list of sub-prefixes
            is_structured = any("R10m/" in p["Key"] for p in img_data_prefixes)

            for band in bands:
                res = S2_RESOLUTION[band]
                if is_structured:
                    # structured format (R10m, R20m, R60m)
                    res_dir = f"R{res}m/"
                    # search path includes the resolution folder
                    search_prefix = f"{img_data_prefix}{res_dir}"

                    # list the files inside the resolution folder
                    band_files = _get_s3_subfolders(
                        s3_client, search_prefix, selectionkey="Contents"
                    )
                    if not band_files:
                        logger.info(f"Access to {search_prefix} failed. Skipping .SAFE file.")
                        continue
                    else:
                        band_files = cast(list[dict], band_files)
                    # filter for specific band file (T..._B04.jp2)
                    image_found = [
                        file["Key"] for file in band_files if f"_B{band}.jp2" in file["Key"]
                    ]
                    if image_found:
                        row["bandImage_{0}".format(band)] = image_found[0]
                        image_found = []

                else:
                    # FLAT FORMAT (Older products, or error fallback)
                    search_prefix = img_data_prefix

                    # List files in IMG_DATA
                    band_files = cast(
                        list, _get_s3_subfolders(s3_client, search_prefix, selectionkey="Contents")
                    )

                    image_found = [
                        file["Key"] for file in band_files if f"_B{band}.jp2" in file["Key"]
                    ]

                    if image_found:
                        row["bandImage_{0}".format(band)] = image_found[0]
                        image_found = []

            row_df = row.to_frame().T
            if image_update.empty:
                image_update = row_df
            else:
                image_update = pd.concat([image_update, row_df], ignore_index=True)

    return image_update


def merge_safe_files(
    satellite: Literal["S1", "S2"],
    bands: list[str],
    output_dir: Path,
    workers: int,
    source: Literal["eodata", "s3"] = "s3",
    local_dir: Path | None = None,
) -> None:
    """Copy all relevant .SAFE files to local directory and acquire spectral band paths.

    Args:
        satellite: S1 for Sentinel-1 and S2 for Sentinel-2.
        bands: (Sub-)set of Sentinel-1 (radar) or Sentinel-2 (spectral) bands.
        output_dir: Directory where lists of required .SAFE files (per parcel id) are stored and
            where to save the output files to.
        workers: Maximum number of workers to use for multiprocessing.
        source: Source of the Sentinel tiles. Either directory ('eodata') or S3 bucket ('s3')
        local_dir: Local directory where the .SAFE files are copied to.
            If None, .SAFE files will not be stored on local disk.

    """

    safe_df = pd.read_pickle(output_dir.joinpath("collector", "full_safe_file_list.pkl"))

    if local_dir is not None:
        local_dir = cast(Path, local_dir)
    # list of unique .SAFE files identifiers
    full_safe_files: pd.Series = safe_df["productIdentifier"]

    if local_dir is not None:
        # Copying the .SAFE files to a local directory massively fastens up the process of opening
        # them later on. Furthermore, opening them directly on the external directory sometimes led
        # to the directory disconnecting from the VM. The same happend for the S3 connection.
        logger.info("Copying files to local storage.")
        max_workers = min(mp_orig.cpu_count(), max(1, min(len(full_safe_files), workers)))
        # for row in full_safe_files.items():
        #     _copy_to_local_dir(source, local_dir, row)
        with mp_orig.Pool(processes=max_workers) as p:
            func = partial(_copy_to_local_dir, source, local_dir)
            process_iter = p.imap(func, full_safe_files.items())
            ti = tqdm(total=len(full_safe_files), desc="Copying .SAFE files to local disk.")
            _ = [ti.update(n=1) for _ in process_iter]
            ti.close()

        logger.info(f"Finished copying all files to local directory {local_dir}.")

        local_safe_files: list[str] = [
            str(local_dir.joinpath(file.lstrip("/"))) for file in full_safe_files
        ]
        safe_files_df: pd.DataFrame = pd.DataFrame(local_safe_files, columns=["productIdentifier"])
        source = "eodata"
    else:
        safe_files_df = pd.DataFrame(full_safe_files.tolist(), columns=["productIdentifier"])

    copier_path: Path = output_dir.joinpath("copier")
    band_path: Path = copier_path.joinpath("band_images.pkl")

    # Collecting all .jp2-paths for each .SAFE file.
    if band_path.is_file():
        # only process the ones that are not already processed.
        band_images_exist: pd.DataFrame = pd.read_pickle(band_path)
        remove_rows: pd.Series[bool] = safe_files_df["productIdentifier"].isin(
            band_images_exist["productIdentifier"]
        )
        safe_files_df.drop(safe_files_df[remove_rows].index, inplace=True)
        if not safe_files_df.empty:
            new_band_images: pd.DataFrame = _get_image_files(
                safe_files_df,
                satellite,
                bands,
                source,
                str(local_dir) if local_dir is not None else "",
            )
            band_images: pd.DataFrame = pd.concat([band_images_exist, new_band_images])
            band_images.to_pickle(copier_path.joinpath("band_images.pkl"))
            logger.info(f"Saved band images to {copier_path.joinpath('band_images.pkl')}.")
    else:
        copier_path.mkdir(exist_ok=True, parents=True)
        band_images = _get_image_files(
            safe_files_df, satellite, bands, source, str(local_dir) if local_dir is not None else ""
        )

        band_images.to_pickle(copier_path.joinpath("band_images.pkl"))
        logger.info(f"Saved band images to {copier_path.joinpath('band_images.pkl')}.")
