"""Dataset sub-command entrypoint for the eurocropsml command-line interface."""

import json
import logging
from pathlib import Path
from typing import Optional, Type, TypeVar

import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eurocropsml.dataset.config import EuroCropsConfig
from eurocropsml.dataset.preprocess import download_dataset, preprocess
from eurocropsml.dataset.splits import create_splits
from eurocropsml.settings import Settings

logger = logging.getLogger(__name__)

datasets_app = typer.Typer(name="datasets")

ConfigT = TypeVar("ConfigT", bound=EuroCropsConfig)
OverridesT = Optional[list[str]]


def build_dataset_app(dataset_name: str, config_class: Type[ConfigT]) -> typer.Typer:
    """Build cli component for preparing datasets."""

    app = typer.Typer(name=dataset_name)

    def build_config(overrides: OverridesT, config_path: str | None = None) -> EuroCropsConfig:
        if config_path is not None:
            config_dir = Path(config_path)
        else:
            config_dir = Settings().cfg_dir.joinpath("dataset")
        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
            if overrides is None:
                overrides = []
            composed_config = compose(config_name="config", overrides=overrides)
        config = config_class(**OmegaConf.to_object(composed_config))  # type: ignore[arg-type]
        return config

    @app.command(name="config")
    def print_config(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to preprocess config"),
    ) -> None:
        """Print currently used config."""
        config = build_config(overrides)
        print(OmegaConf.to_yaml(json.loads(config.model_dump_json())))

    @app.command(name="download")
    def download_data(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to preprocess config"),
    ) -> None:
        config = build_config(overrides, config_path)

        download_dataset(
            preprocess_config=config.preprocess,
        )

    @app.command(name="preprocess")
    def preprocess_data(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to preprocess config"),
    ) -> None:
        config = build_config(overrides, config_path)
        preprocess(
            preprocess_config=config.preprocess,
        )

    @app.command()
    def build_splits(
        config_path: str = typer.Option(None, "--config-path", help="Path to config.yaml file."),
        overrides: OverridesT = typer.Argument(None, help="Overrides to split config"),
    ) -> None:
        config = build_config(overrides, config_path)
        create_splits(config.split, config.preprocess.raw_data_dir.parent)

    return app


datasets_app.add_typer(build_dataset_app("eurocrops", EuroCropsConfig))
