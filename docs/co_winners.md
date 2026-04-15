# Égalité  

On compute les égalités comme suit 

```py

def _compute_cowinners(rule: object) -> list[str]:
    profile = getattr(rule, "profile_", None)
    weak_winners = getattr(profile, "weak_condorcet_winners", None)
    if weak_winners is not None:
        weak_winner_indices = np.flatnonzero(np.asarray(weak_winners, dtype=bool))
        n_candidates = getattr(profile, "n_c", None)
        if isinstance(n_candidates, int) and n_candidates == 2 and weak_winner_indices.size == 2:
            return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in weak_winner_indices]

    scores = getattr(rule, "scores_", None)
    if scores is not None:
        try:
            scores_array = np.asarray(scores)
        except Exception:
            scores_array = None
        if scores_array is not None and scores_array.ndim == 1 and scores_array.size > 0:
            best_score = np.max(scores_array)
            winner_indices = np.flatnonzero(np.isclose(scores_array, best_score, equal_nan=False))
            if winner_indices.size > 0:
                return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in winner_indices]

    winner_index = _winner_index(rule)
    if winner_index is not None:
        return [_label_for_candidate(rule, winner_index)]

    if weak_winners is not None:
        weak_winner_indices = np.flatnonzero(np.asarray(weak_winners, dtype=bool))
        if weak_winner_indices.size > 0:
            return [_label_for_candidate(rule, int(candidate_index)) for candidate_index in weak_winner_indices]
    return []


def _ensure_cowinners(rule: Any) -> RuleResult:
    if hasattr(rule, "cowinners_"):
        return rule
    rule.cowinners_ = _compute_cowinners(rule)
    return rule
```


