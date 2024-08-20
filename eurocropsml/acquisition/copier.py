"""Copying files to local directories for faster processing."""

import logging
import multiprocessing as mp_orig
import os
import shutil
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _copy_to_local_dir(local_dir: Path, safe_file: str) -> None:
    """Copying files to local directory.

    Args:
        local_dir: Directory to copy the file to.
        safe_file: File to copy to local directory.

    """
    # Copy all image files from network storage over to local storage
    filename = safe_file[1]
    local_product: Path = local_dir.joinpath(filename[1:])
    local_parent_dir: Path = local_product.parents[0]
    if not local_parent_dir.exists():
        local_parent_dir.mkdir(exist_ok=True, parents=True)

    if not local_product.exists():
        shutil.copytree(filename, local_product)


def _get_image_files(full_safe_files: pd.DataFrame, spectral_bands: list[str]) -> pd.DataFrame:
    """Getting paths for each spectral band.

    Args:
        full_safe_files: DataFrame with safe file paths for which to get the band paths.
        spectral_bands: Spectral bands to consider.

    Returns:
        DataFrame with band paths as columns.

    """

    image_update = pd.DataFrame(columns=full_safe_files.columns.values)
    for idx, row in tqdm(
        full_safe_files.iterrows(),
        total=len(full_safe_files),
        desc="Collecting paths for spectral bands.",
    ):
        if Path(row["productIdentifier"]).exists():
            filename_list: list = os.listdir(os.path.join(row["productIdentifier"], "GRANULE"))
            filename: str = filename_list[0]
            sub_path: str = os.path.join("GRANULE", filename, "IMG_DATA")
            path_files: str = os.path.join(row["productIdentifier"], sub_path)
            resolutions: list[int] = [10, 20, 60]

            files: list[str] = os.listdir(path_files)

            for band in spectral_bands:
                if "R10m" in files:
                    for res in resolutions:
                        r1_files: list[str] = os.listdir(
                            os.path.join(path_files, "R{0}m".format(res))
                        )
                        image_found: list[str] = [file for file in r1_files if f"_B{band}" in file]
                        if image_found:
                            row["bandImage_{0}".format(band)] = os.path.join(
                                sub_path, "R{0}m".format(res), image_found[0]
                            )
                            image_found = []
                            break

                else:
                    image_found = [file for file in files if f"_B{band}" in file]
                    row["bandImage_{0}".format(band)] = os.path.join(path_files, image_found[0])
                    image_found = []

            if idx == 0:
                image_update = row.to_frame()
                image_update = image_update.T

            else:
                row_df: pd.DataFrame = row.to_frame()
                row_df = row_df.T
                image_update = pd.concat([image_update, row_df], ignore_index=True)

    return image_update


def merge_s2_safe_files(
    bands: list[str],
    output_dir: Path,
    workers: int,
    local_dir: Path | None = None,
) -> None:
    """Copy all relevant safe files to local directory and acquire spectral band paths.

    Args:
        bands: Sentinel-2 bands.
        output_dir: Directory where lists of required SAFE-files (per parcel id) are stored and
            where to save the output files to.
        workers: Maximum number of workers to use for multiprocessing.
        local_dir: Local directory where the SAFE-files are copied to.
            If None, SAFE-files will not be stored on local disk.

    """

    safe_df = pd.read_pickle(output_dir.joinpath("collector", "full_safe_file_list.pkg"))

    # list of unique .SAFE-files identifiers
    full_safe_files: pd.Series = safe_df["productIdentifier"]

    if local_dir is not None:
        # Copying the .SAFE-files to a local directory massively fastens up the process of opening
        # them later on. Furthermore, opening them directly on the external directory sometimes led
        # to the directory disconnecting from the VM.
        local_dir = cast(Path, local_dir)
        logger.info("Copying files to local storage.")
        max_workers = min(mp_orig.cpu_count(), max(1, min(len(full_safe_files), workers)))
        with mp_orig.Pool(processes=max_workers) as p:
            func = partial(_copy_to_local_dir, local_dir)
            process_iter = p.imap(func, full_safe_files.items())
            ti = tqdm(total=len(full_safe_files), desc="Copying SAFE-files to local disk.")
            _ = [ti.update(n=1) for _ in process_iter]
            ti.close()

        logger.info(f"Finished copying all files to local directory {local_dir}.")

        local_safe_files: list[str] = [
            str(local_dir.joinpath(file[1:])) for file in full_safe_files
        ]
        safe_files_df: pd.DataFrame = pd.DataFrame(local_safe_files, columns=["productIdentifier"])
    else:
        safe_files_df = pd.DataFrame(full_safe_files.tolist(), columns=["productIdentifier"])

    copier_path: Path = output_dir.joinpath("copier")
    band_path: Path = copier_path.joinpath("band_images.pkg")

    # Collecting all .jp2-paths for each .SAFE-file.
    if band_path.is_file():
        # only process the ones that are not already processed.
        band_images_exist: pd.DataFrame = pd.read_pickle(band_path)
        remove_rows: pd.Series[bool] = safe_files_df["productIdentifier"].isin(
            band_images_exist["productIdentifier"]
        )
        safe_files_df.drop(safe_files_df[remove_rows].index, inplace=True)
        new_band_images: pd.DataFrame = _get_image_files(safe_files_df, bands)
        band_images: pd.DataFrame = pd.concat([band_images_exist, new_band_images])
    else:
        copier_path.mkdir(exist_ok=True, parents=True)
        band_images = _get_image_files(safe_files_df, bands)

    band_images.to_pickle(copier_path.joinpath("band_images.pkg"))
    logger.info(f"Saved band images to {copier_path.joinpath('band_images.pkg')}.")
