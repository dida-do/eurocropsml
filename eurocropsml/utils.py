"""Auxiliary module for global utils functions."""

import hashlib
import shutil
import zipfile
from pathlib import Path


def _move_files(src_dir: Path, dest_dir: Path) -> None:
    """Move files from src_dir to dest_dir."""
    dest_dir.mkdir(exist_ok=True, parents=True)

    for item in src_dir.iterdir():
        dest_item = dest_dir.joinpath(item.name)

        if item.is_file():
            shutil.move(item, dest_item)


def _create_md5_hash(filepath: Path) -> str:
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _unzip_file(zip_filepath: Path, extract_to_path: Path, delete_zip: bool = True) -> None:
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)
    # delete zip-file
    if delete_zip:
        zip_filepath.unlink()
