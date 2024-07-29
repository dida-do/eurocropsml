"""Auxiliary module for global utils functions."""

import zipfile
from pathlib import Path


def _unzip_file(zip_filepath: Path, extract_to_path: Path) -> None:
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)
