import matplotlib
import numpy as np

from vote_simulation.models.simulation_result import SimulationSeriesResult, SimulationStepResult

matplotlib.use("Agg")


def test_add_method_result_builds_symmetric_uint8_matrix() -> None:
    step = SimulationStepResult(data_source="sample.parquet")

    step.add_method_result("borda", ["A"])
    step.add_method_result("stv", ["A"])
    step.add_method_result("irv", ["B"])

    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )

    assert step.rule_codes == ["BORDA", "STV", "IRV"]
    assert step.dist_matrix.dtype == np.float32
    assert np.array_equal(step.dist_matrix, expected)


def test_updating_existing_rule_refreshes_only_its_distances() -> None:
    step = SimulationStepResult(
        data_source="sample.parquet",
        winners_by_rule={
            "BORDA": ["A"],
            "STV": ["A"],
            "IRV": ["B"],
        },
    )

    step.add_method_result("stv", ["B"])

    expected = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(step.dist_matrix, expected)


def test_string_representation_contains_pretty_matrix() -> None:
    step = SimulationStepResult(data_source="sample.parquet")
    step.add_method_result("borda", ["A"])
    step.add_method_result("stv", ["B"])

    rendered = str(step)

    assert "Data Source: sample.parquet" in rendered
    assert "Winners by rule:" in rendered
    assert "Distance Matrix:" in rendered
    assert "BORDA" in rendered
    assert "STV" in rendered


def test_plot_distance_matrix_returns_axes() -> None:
    step = SimulationStepResult(data_source="sample.parquet")
    step.add_method_result("borda", ["A"])
    step.add_method_result("stv", ["B"])

    ax = step.plot_distance_matrix(show=False)

    assert "Rule distance matrix" in ax.get_title()
    assert ax.get_aspect() == 1.0
    assert [tick.get_text() for tick in ax.get_xticklabels()[:2]] == ["BORDA", "STV"]


def test_series_step_count_and_mean_diagonal_is_zero() -> None:
    """Mean of binary matrices should have a zero diagonal."""

    series = SimulationSeriesResult()
    for _ in range(5):
        step = SimulationStepResult(data_source="iter.parquet")
        step.add_method_result("BORDA", ["A"])
        step.add_method_result("STV", ["B"])
        series.add_step(step)

    assert series.step_count == 5
    mean = series.mean_distance_matrix
    assert mean.dtype == np.float32
    assert np.all(np.diag(mean) == 0)


def test_series_mean_values_correct() -> None:
    """Mean matrix should average properly over iterations."""

    series = SimulationSeriesResult()

    # 3 steps where BORDA and STV always disagree → mean distance = 1.0
    step_always = SimulationStepResult(data_source="a.parquet")
    step_always.add_method_result("BORDA", ["A"])
    step_always.add_method_result("STV", ["B"])
    series.add_step(step_always)
    series.add_step(step_always)
    series.add_step(step_always)

    # 1 step where they agree → mean distance = 0.75 over 4 steps
    step_agree = SimulationStepResult(data_source="b.parquet")
    step_agree.add_method_result("BORDA", ["A"])
    step_agree.add_method_result("STV", ["A"])
    series.add_step(step_agree)

    mean = series.mean_distance_matrix
    borda_idx = series._rule_index["BORDA"]
    stv_idx = series._rule_index["STV"]
    # 3/4 steps disagree → 75 on the [0,100] scale
    assert abs(float(mean[borda_idx, stv_idx]) - 75.0) < 0.1


def test_series_plot_mean_distance_matrix_returns_axes() -> None:
    series = SimulationSeriesResult()
    step = SimulationStepResult(data_source="x.parquet")
    step.add_method_result("BORDA", ["A"])
    step.add_method_result("STV", ["B"])
    series.add_step(step)

    ax = series.plot_mean_distance_matrix(show=False)

    assert "Mean rule distance matrix" in ax.get_title()
    assert ax.get_aspect() == 1.0
