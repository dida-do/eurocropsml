"""Main entrypoint for the eurocropsml command-line interface."""

import logging

import typer

from eurocropsml.acquisition.cli import acquisition_app
from eurocropsml.dataset.cli import datasets_app

logger = logging.getLogger(__name__)

cli = typer.Typer(name="EuroCrops")


@cli.callback()
def logging_setup() -> None:
    """Logging setup for CLI."""

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )


cli.add_typer(acquisition_app)
cli.add_typer(datasets_app)

if __name__ == "__main__":
    cli()
