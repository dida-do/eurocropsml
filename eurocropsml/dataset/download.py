"""Download utilities for the EuroCropsML dataset Zenodo record."""

import logging
import shutil
import sys
from pathlib import Path
from typing import cast

import requests
import typer

from eurocropsml.dataset.config import EuroCropsDatasetPreprocessConfig
from eurocropsml.utils import _create_md5_hash, _move_files, _unzip_file

logger = logging.getLogger(__name__)


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
        logger.info(f"Downloading {local_path}...")
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


def get_user_choice() -> list[str]:
    """Get user choice for which files to download."""
    choice = input(
        "Would you like to download Sentinel-1 and/or Sentinel-2 data? Please enter "
        " 'S1', 'S2', or 'both': "
    )
    if choice not in {"S1", "S2", "both"}:
        logger.error("Invalid input. Please enter 'S1', 'S2', or 'both'.")
        sys.exit(1)
    elif choice == "both":
        choice_list = ["S1", "S2"]
        logger.info("Downloading both S1 and S2 data.")
    else:
        logger.info(f"Downloading only {choice} data.")
        choice_list = [choice]

    return choice_list


def select_version(versions: list[dict]) -> tuple[dict, list[str]]:
    """Select one of the available dataset versions on Zenodo."""
    # we only keep version 4 (published Mar 14, 2024) and newer
    # since older versions contain incorrect data
    filtered_versions: list[dict] = [
        version for version in versions if version["metadata"]["publication_date"] >= "2024-03-14"
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
        if selected_id >= 7:
            logger.warning(
                "Please be aware that the folder structure of version 7 or older is not "
                "supported in this package version and you need to manually move the files"
                " after downloading as follows\n"
                "\n"
                "path/to/data_dir"
                "    ├── preprocess/"
                "    │   └── S2/"
                "    │       ├── <NUTS3>_<parcel_id>_<EC_hcat_c>.npz"
                "    │       └── ..."
                "    └── raw_data/"
                "        ├── geometries/"
                "        │   ├── Estonia.geojson"
                "        │   ├── Latvia.geojson"
                "        │   └── Portugal.geojson"
                "        ├── labels/"
                "        │   ├── Estonia_labels.parquet"
                "        │   ├── Latvia_labels.parquet"
                "        │   └── Portugal_labels.parquet"
                "        └── S2/"
                "            ├── Estonia.parquet"
                "            ├── Latvia.parquet"
                "            └── Portugal.parquet"
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

    response: requests.models.Response = requests.get(base_url)

    try:
        response.raise_for_status()
        data = response.json()
        versions: list[dict] = data["hits"]["hits"]

        if versions:
            selected_version, files_to_download = select_version(versions)

            # older version do only have S2 data
            # if S1 data is available, let user decide
            if "S1.zip" in files_to_download:
                user_choice = get_user_choice()
                if "S1" not in user_choice:
                    files_to_download.remove("S1.zip")
                if "S2" not in user_choice:
                    files_to_download.remove("S2.zip")

            for file_entry in selected_version["files"]:
                file_url: str = file_entry["links"]["self"]
                zip_file: str = file_entry["key"]
                if zip_file in files_to_download:
                    local_path: Path = data_dir.joinpath(zip_file)
                    _download_file(zip_file, file_url, local_path, file_entry.get("checksum", ""))
                    logger.info(f"Unzipping {local_path}...")
                    _unzip_file(local_path, data_dir)

                    # move S1 and S2 data
                    if zip_file in ["S1.zip", "S2.zip"]:
                        unzipped_path: Path = local_path.with_suffix("")
                        for folder in unzipped_path.iterdir():
                            rel_target_folder: Path = folder.relative_to(unzipped_path)
                            _move_files(
                                folder, data_dir.joinpath(rel_target_folder, zip_file.split(".")[0])
                            )
                        shutil.rmtree(unzipped_path)
        else:
            logger.error("No data found on Zenodo. Please download manually.")
            sys.exit(1)

    except requests.exceptions.HTTPError as err:
        logger.warning(f"There was an error when trying to access the Zenodo record: {err}")