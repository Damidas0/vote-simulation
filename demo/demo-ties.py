import numpy as np
from svvamp import Profile

from vote_simulation.models.rules.registry import get_all_rules_codes
from vote_simulation.simulation.simulation import simulation_step

if __name__ == "__main__":
    ballots = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    profile = Profile(preferences_ut=ballots, labels_candidates=["A", "B", "C"])

    candidates = ["A", "B", "C"]

    rules_codes = get_all_rules_codes()

    result = simulation_step(profile, rules_codes)
    print(result)
    for rule_code, rule_result in result.winners_by_rule.items():
        # Skip L4VD which is not implemented
        if rule_code == "L4VD":
            continue
        assert rule_result == ["A", "B", "C"], f"Rule {rule_code} did not handle ties correctly"
