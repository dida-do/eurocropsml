"""Auxiliary module for global utils functions."""

import hashlib
import shutil
import zipfile
from pathlib import Path

import typer


def _move_(src_dir: Path, dest_dir: Path) -> None:
    """Move files or folders from src_dir to dest_dir."""

    move_files: bool = True
    if dest_dir.exists() and _compare_folders(src_dir, dest_dir):
        move_files = typer.confirm(
            f"{dest_dir} already exists and the content is different from the new data. Do you"
            " want to replace the existing files?"
        )
    if move_files:
        dest_dir.mkdir(exist_ok=True, parents=True)

        for item in src_dir.iterdir():
            dest_item = dest_dir.joinpath(item.name)

            shutil.move(item, dest_item)


def _create_md5_hash(filepath: Path) -> str:
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _hash_folder(folder_path: Path) -> str:
    """Compute a combined hash for all files in a folder."""
    hash_obj = hashlib.md5()

    for file_path in sorted(folder_path.rglob("*")):
        # hash only files
        if file_path.is_file():
            file_hash = _create_md5_hash(file_path)
            # Update folder hash with the relative path and file hash for consistency
            relative_path = file_path.relative_to(folder_path)
            hash_obj.update(str(relative_path).encode())
            hash_obj.update(file_hash.encode())

    return hash_obj.hexdigest()


def _compare_folders(folder1: Path, folder2: Path) -> bool:
    """Compare two folders by their combined hash values."""
    hash_folder1 = _hash_folder(folder1)
    hash_folder2 = _hash_folder(folder2)
    return hash_folder1 == hash_folder2


def _unzip_file(zip_filepath: Path, extract_to_path: Path, delete_zip: bool = True) -> None:
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)
    # delete zip-file
    if delete_zip:
        zip_filepath.unlink()
