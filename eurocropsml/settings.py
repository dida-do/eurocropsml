"""Auxiliary module for handling global package settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parents[1]


class Settings(BaseSettings):
    """Global settings."""

    cfg_dir: Path = Field(Path("eurocropsml", "configs"), validation_alias="EUROCROPS_CONFIG_DIR")
    data_dir: Path = Field(Path("data"), validation_alias="EUROCROPS_DATA_DIR")
    acquisition_dir: Path = Field(Path("acquisition"), validation_alias="EUROCROPS_ACQUISITION_DIR")

    @field_validator("cfg_dir", "data_dir", "acquisition_dir")
    @classmethod
    @classmethod
    def relative_path(cls, v: Path) -> Path:
        """Interpret relative paths w.r.t. the project root."""
        if not v.is_absolute():
            v = ROOT_DIR.joinpath(v)
        return v
