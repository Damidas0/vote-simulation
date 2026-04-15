from pathlib import Path

import pytest

from vote_simulation.simulation.configuration import load_simulation_config


def test_load_simulation_config_missing_simulation_section(tmp_path: Path):
    """Missing [simulation] section raises an explicit error."""

    config_file = tmp_path / "simulation.toml"
    config_file.write_text('title = "oops"', encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing \[simulation\] section"):
        load_simulation_config(config_file)
