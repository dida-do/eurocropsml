from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel
from typer.testing import CliRunner, Typer

from eurocropsml.dataset.cli import build_dataset_app


class MockConfig(BaseModel):
    name: str
    value: int


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


@pytest.fixture
def config(monkeypatch: Any, tmp_path: Path) -> MockConfig:

    config = MockConfig(name="test_dataset", value=1)

    data_path = tmp_path / "eurocropsml" / "configs"
    config_path = data_path / "dataset" / "config.yaml"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config.model_dump()))

    monkeypatch.setenv("EUROCROPS_CONFIG_DIR", str(data_path.absolute()))

    return config


@pytest.fixture
def app(config: MockConfig) -> Typer:
    return build_dataset_app(
        dataset_name=config.name,
        config_class=MockConfig,  # type: ignore[type-var]
    )


def test_print_config(app: Typer, runner: CliRunner, config: MockConfig) -> None:
    result = runner.invoke(app, ["config"], catch_exceptions=False)

    assert result.exit_code == 0
    assert result.stdout.strip() == OmegaConf.to_yaml(config.model_dump()).strip()


def test_print_config_overrides(app: Typer, runner: CliRunner, config: MockConfig) -> None:
    result = runner.invoke(app, ["config", "value=2"], catch_exceptions=False)
    config.value = 2
    assert result.exit_code == 0
    assert result.stdout.strip() == OmegaConf.to_yaml(config.model_dump()).strip()
