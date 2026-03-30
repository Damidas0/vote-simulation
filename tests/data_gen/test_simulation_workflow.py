"""Tests for the full simulation workflow with data generation."""

from pathlib import Path

import pytest

from vote_simulation.simulation.configuration import load_simulation_config
from vote_simulation.simulation.simulation import (
    generate_data,
    obtain_data_instance,
    simulation_full,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ Config parsing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfigGenerativeModels:
    """Configuration correctly parses generative_models and related fields."""

    def test_parse_generative_models(self, tmp_path: Path):
        cfg = tmp_path / "sim.toml"
        cfg.write_text(
            """
            [simulation]
            rule_codes = ["PLU1"]
            generative_models = ["UNI", "ic"]
            candidates = [3]
            voters = [9]
            iterations = 2
            seed = 42
            """.strip(),
            encoding="utf-8",
        )
        config = load_simulation_config(cfg)
        assert config.generative_models == ["UNI", "IC"]
        assert config.iterations == 2
        assert config.seed == 42

    def test_generator_params_section(self, tmp_path: Path):
        cfg = tmp_path / "sim.toml"
        cfg.write_text(
            """
            [simulation]
            rule_codes = ["PLU1"]
            generative_models = ["EUCLID"]
            candidates = [3]
            voters = [9]
            iterations = 1
            seed = 0

            [generator_params.EUCLID]
            box_dimensions = [1.0, 2.0]
            """.strip(),
            encoding="utf-8",
        )
        config = load_simulation_config(cfg)
        assert "EUCLID" in config.generator_params
        assert config.generator_params["EUCLID"]["box_dimensions"] == [1.0, 2.0]

    def test_missing_generative_models_defaults_empty(self, tmp_path: Path):
        cfg = tmp_path / "sim.toml"
        cfg.write_text(
            """
            [simulation]
            rule_codes = ["PLU1"]
            """.strip(),
            encoding="utf-8",
        )
        config = load_simulation_config(cfg)
        assert config.generative_models == []


# ━━━━━━━━━━━━━━━━━━━━━━━ obtain_data_instance ━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestObtainDataInstance:
    """Test the generate-or-cache logic."""

    def test_generates_and_caches(self, tmp_path: Path):
        base = str(tmp_path)
        di = obtain_data_instance("UNI", 10, 3, iteration=0, seed=42, base_path=base)

        # File should have been created
        expected = tmp_path / "gen" / "UNI_v10_c3" / "iter_0001.parquet"
        assert expected.is_file()
        assert di.file_path == str(expected)
        assert di.n_voters == 10
        assert di.n_candidates == 3

    def test_loads_from_cache(self, tmp_path: Path):
        base = str(tmp_path)
        di1 = obtain_data_instance("IC", 5, 3, iteration=0, seed=7, base_path=base)
        di2 = obtain_data_instance("IC", 5, 3, iteration=0, seed=7, base_path=base)
        # Both should point to the same file
        assert di1.file_path == di2.file_path
        import numpy as np

        np.testing.assert_array_almost_equal(di1.data, di2.data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━ generate_data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGenerateData:
    """Test the bulk generation entry-point."""

    def test_generates_expected_count(self, tmp_path: Path):
        cfg = tmp_path / "sim.toml"
        cfg.write_text(
            f"""
            [simulation]
            rule_codes = ["PLU1"]
            generative_models = ["UNI"]
            candidates = [3, 4]
            voters = [9, 15]
            iterations = 2
            seed = 0
            output_base_path = "{tmp_path}/out"
            """.strip(),
            encoding="utf-8",
        )
        paths = generate_data(str(cfg))
        # 1 model × 2 candidates × 2 voters × 2 iterations = 8
        assert len(paths) == 8
        for p in paths:
            assert Path(p).is_file()

    def test_raises_without_models(self, tmp_path: Path):
        cfg = tmp_path / "sim.toml"
        cfg.write_text(
            """
            [simulation]
            rule_codes = ["PLU1"]
            candidates = [3]
            voters = [9]
            iterations = 1
            seed = 0
            """.strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="generative_models"):
            generate_data(str(cfg))


# ━━━━━━━━━━━━━━━━━━━━━━━━ simulation_full ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSimulationFull:
    """Integration test: generate → run rules → save results."""

    def test_full_pipeline_small(self, tmp_path: Path):
        cfg = tmp_path / "sim.toml"
        cfg.write_text(
            f"""
            [simulation]
            rule_codes = ["PLU1", "BORD"]
            generative_models = ["UNI"]
            candidates = [3]
            voters = [9]
            iterations = 2
            seed = 42
            output_base_path = "{tmp_path}/out"
            """.strip(),
            encoding="utf-8",
        )
        simulation_full(str(cfg))

        # Check gen files
        gen_dir = tmp_path / "out" / "gen" / "UNI_v9_c3"
        assert (gen_dir / "iter_0001.parquet").is_file()
        assert (gen_dir / "iter_0002.parquet").is_file()

        # Check sim_result files
        sim_dir = tmp_path / "out" / "sim_result" / "UNI_v9_c3"
        assert (sim_dir / "iter_0001.parquet").is_file()
        assert (sim_dir / "iter_0002.parquet").is_file()
