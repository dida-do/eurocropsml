"""Acquisition sub-command entrypoint for the eurocropsml command-line interface."""

import json
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar

import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eurocropsml.acquisition.build import build_dataset
from eurocropsml.acquisition.config import Config
from eurocropsml.settings import Settings

logger = logging.getLogger(__name__)

acquisition_app = typer.Typer(name="acquisition")

ConfigT = TypeVar("ConfigT", bound=Config)
OverridesT = Optional[list[str]]


def build_aquisition_app(dataset_name: str, config_class: Type[ConfigT]) -> typer.Typer:
    """Build cli component for acquiring reflectance data."""

    app = typer.Typer(name=dataset_name)

    def build_config(overrides: OverridesT, config_path: str | None = None) -> Config:
        if config_path is not None:
            config_dir = Path(config_path)
        else:
            config_dir = Settings().cfg_dir.joinpath("acquisition")
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            if overrides is None:
                overrides = []
            composed_config = compose(config_name="config", overrides=overrides)
        config = config_class(**OmegaConf.to_object(composed_config))  # type: ignore[arg-type]
        return config

    @app.command(name="config")
    def print_config(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to config"),
    ) -> None:
        """Print currently used config."""
        config = build_config(overrides)
        print(OmegaConf.to_yaml(json.loads(config.cfg.model_dump_json())))

    @app.command()
    def get_data(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to config"),
    ) -> None:

        config = build_config(overrides, config_path)

        build_dataset(config.cfg)

    return app


acquisition_app.add_typer(build_aquisition_app("eurocrops", Config))
