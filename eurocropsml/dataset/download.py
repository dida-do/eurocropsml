"""Download utilities for the EuroCropsML dataset Zenodo record."""

import logging
import shutil
import sys
from pathlib import Path
from typing import cast

import requests
import typer

from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.utils import _create_md5_hash, _move_, _unzip_file

logger = logging.getLogger(__name__)


def _get_zenodo_record(
    base_url: str, version_number: int | None = None
) -> tuple[dict, list[str]] | dict:
    response: requests.models.Response = requests.get(base_url)
    response.raise_for_status()
    data = response.json()
    versions: list[dict] = data["hits"]["hits"]

    if versions:
        if version_number is not None:
            selected_version = next(
                (
                    v
                    for v in versions
                    if v["metadata"]["relations"]["version"][0]["index"] + 1 == version_number
                ),
                None,
            )
            if selected_version is not None:
                return selected_version
            else:
                logger.error(f"Version {version_number} could not be found on Zenodo.")
                sys.exit(1)
        else:
            selected_version, files_to_download = select_version(versions)
            return selected_version, files_to_download
    else:
        logger.error("No data found on Zenodo. Please download manually.")
        sys.exit(1)


def _download_file(
    file_name: str, file_url: str, local_path: Path, downloadfile_md5_hash: str
) -> None:
    """Download a file from a URL if it doesn't already exist."""
    download: bool = True
    if local_path.exists():
        local_md5 = _create_md5_hash(local_path)
        if downloadfile_md5_hash in [local_md5, "md5:" + local_md5]:
            logger.info(f"{local_path} already exists. Skip downloading.")
            return
        else:
            download = typer.confirm(
                f"{local_path} already exists but with different data."
                " Do you want to delete it and redownload the file?"
            )

    if download:
        logger.info(f"Downloading to {local_path}...")
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            with open(local_path, "wb") as file:
                file.write(response.content)
            logger.info(f"{file_name} downloaded.")
        except requests.exceptions.HTTPError as err:
            logger.warning(
                f"There was an error when trying to get the Zenodo record {file_url}: {err}"
            )

    else:
        logger.info(f"{local_path} will not be downloaded again.")


def get_user_choice(files_to_download: list[str]) -> list[str]:
    """Get user choice for which files to download."""
    logger.info("Choose one or more of the following options by typing their numbers (e.g., 1 3):")
    for i, file in enumerate(files_to_download, 1):
        logger.info(f"{i}. {file}")
    choice = typer.prompt("Enter your choices separated by spaces")
    selected_indices = [int(choice) - 1 for choice in choice.split()]
    selected_options = [files_to_download[i] for i in selected_indices]

    return selected_options


def select_version(versions: list[dict]) -> tuple[dict, list[str]]:
    """Select one of the available dataset versions on Zenodo."""
    # we only keep version 4 (published Mar 14, 2024) and newer
    # since older versions contain incorrect data. We also remove version 9 and 10
    # since they are either missing data (v9) or have corrupted files (v10).
    filtered_versions: list[dict] = [
        version
        for version in versions
        if version["metadata"]["publication_date"] >= "2024-03-14"
        and version["metadata"]["publication_date"] not in ("2025-02-06", "2025-03-18")
    ]
    filtered_versions = sorted(
        filtered_versions, key=lambda version: version["metadata"]["publication_date"]
    )
    logger.info("Available EuroCropsML versions on Zenodo:")
    version_ids: dict[int, list[int | list[str]]] = {}
    for i, version in enumerate(filtered_versions):
        version_id: int = version["metadata"]["relations"]["version"][0]["index"] + 1
        title: str = version["metadata"]["title"]
        version_doi: str = version["links"]["doi"]
        # get available filenames
        file_names: list[str] = [file["key"] for file in version["files"]]
        version_ids[version_id] = [i, file_names]
        logger.info(
            f"v{version_id}: {title} (DOI: {version_doi}; available files: {', '.join(file_names)})"
        )

    # User selects the version
    selected_id: int = int(input("Please select one of the above version numbers: "))
    if selected_id in version_ids:
        logger.info(f"Selected version {selected_id}.")
        if selected_id <= 11:

            logger.warning(
                "Please be aware that the folder structure of Zenodo version 11 or older is not "
                "supported in this package version (eurocropsml>=0.4.0) and you need to manually "
                "rename and move the files after downloading as follows:\n"
                "\n"
                "path/to/data_dir\n"
                "    ├── preprocess/\n"
                "    │   └── S2/\n"
                "    │      └── 2021/\n"
                "    │          ├── <NUTS3>_<parcel_id>_<EC_hcat_c>.npz\n"
                "    │          └── ...\n"
                "    └── raw_data/\n"
                "        ├── geometries/\n"
                "        │   └── 2021/\n"
                "        │      ├── EE.geojson\n"
                "        │      ├── LV.geojson\n"
                "        │      └── PT.geojson\n"
                "        ├── labels/\n"
                "        │   └── 2021/\n"
                "        │      ├── EE_labels.parquet\n"
                "        │      ├── LV_labels.parquet\n"
                "        │      └── PT_labels.parquet\n"
                "        └── S2/\n"
                "            └── 2021/\n"
                "                   ├──allyear"
                "                   ├── EE.parquet\n"
                "                   ├── LV.parquet\n"
                "                   └── PT.parquet\n"
                "    ...\n"
                "\n"
            )
        return filtered_versions[cast(int, version_ids[selected_id][0])], cast(
            list[str], version_ids[selected_id][1]
        )
    else:
        logger.error("Invalid version selection.")
        sys.exit(1)


def download_dataset(preprocess_config: EuroCropsDatasetPreprocessConfig) -> None:
    """Download EuroCropsML dataset from Zenodo.

    Args:
        preprocess_config: Config used to access the Zenodo URL.

    Raises:
        requests.exceptions.HTTPError: If Zenodo records could not be accessed.
    """

    base_url: str = preprocess_config.download_url

    data_dir: Path = Path(preprocess_config.raw_data_dir.parent)
    data_dir.mkdir(exist_ok=True, parents=True)

    try:
        selected_version, files_to_download = _get_zenodo_record(base_url)
        # let user decide what data to download
        selected_files = get_user_choice(files_to_download)
        for file_entry in selected_version["files"]:
            file_url: str = file_entry["links"]["self"]
            zip_file: str = file_entry["key"]
            if zip_file in selected_files:
                local_path: Path = data_dir.joinpath(zip_file)
                _download_file(zip_file, file_url, local_path, file_entry.get("checksum", ""))
                logger.info(f"Unzipping {local_path}...")
                _unzip_file(local_path, data_dir)

                # move S1 and S2 data
                if zip_file in ["S1.zip", "S2.zip"]:
                    unzipped_path: Path = local_path.with_suffix("")
                    for folder in unzipped_path.iterdir():
                        rel_parent_target_folder: Path = folder.relative_to(unzipped_path)
                        for sub_folder in folder.iterdir():
                            rel_target_folder: Path = sub_folder.relative_to(folder)
                            _move_(
                                sub_folder,
                                data_dir.joinpath(
                                    rel_parent_target_folder,
                                    zip_file.split(".")[0],
                                    rel_target_folder,
                                ),
                            )
                    shutil.rmtree(unzipped_path)

    except requests.exceptions.HTTPError as err:
        logger.warning(f"There was an error when trying to access the Zenodo record: {err}")
