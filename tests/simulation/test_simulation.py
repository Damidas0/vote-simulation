import os
from svvamp import Profile
import tempfile

import numpy as np
import pytest

from vote_simulation.models.results.result_config import ResultConfig
from vote_simulation.models.results.series_result import SimulationSeriesResult
from vote_simulation.models.results.step_result import SimulationStepResult
from vote_simulation.models.results.total_result import (
    SeriesKey,
    SimulationTotalResult,
    _extract_key,
)


from vote_simulation.models.rules.registry import get_all_rules_codes
from vote_simulation.simulation.simulation import simulation_step


def test_wrong_file_path():
    """test if a filepath that does not exist raises error"""
    return  # TODO: implement test for wrong file path


def test_wrong_file_format():
    """test for unsupported file format"""
    return  # TODO: implement test for wrong file format

def test_ties_cases():
    """test if ties are properly handled"""

    ballots = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    profile = Profile(preferences_ut=ballots, labels_candidates=["A", "B", "C"])

    rules_codes = get_all_rules_codes()

    result = simulation_step(profile, rules_codes)

    for rule_code, rule_result in result.winners_by_rule.items():
        if rule_code == "L4VD":
            continue
        assert rule_result == ["A", "B", "C"], f"Rule {rule_code} did not handle ties correctly"