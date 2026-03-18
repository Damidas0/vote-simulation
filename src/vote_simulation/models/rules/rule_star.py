from whalrus import NiceDict, Priority, RuleScore, cached_property

class RuleStar(RuleScore):
    """
    STAR rules is a two round voting system. 
    1. Voters score each candidate. 
    2. The two candidates with the highest scores are selected and a plurality vote is held between them.
    """

    def __init__(self, *args, tie_break: Priority = Priority.ASCENDING, **kwargs):
        super().__init__(*args, tie_break=tie_break, **kwargs)

    @cached_property
    def first_round_scores_(self) -> NiceDict:
        """Scores from the first STAR round"""
        return NiceDict({
            candidate: sum(ballot[candidate] for ballot in self.profile_converted_)
            for candidate in self.candidates_
        })

    @cached_property
    def finalists_(self) -> list[str]:
        """Two candidates selected for STAR runoff."""
        ordered_candidates = sorted(
            self.candidates_,
            key=lambda candidate: (self.first_round_scores_[candidate], candidate),
            reverse=True,
        )
        return ordered_candidates[:2]

    @cached_property
    def scores_(self) -> NiceDict:
        """Composite STAR scores: runoff votes first, first-round score as tie-break."""
        if len(self.candidates_) < 2:
            only_candidate = next(iter(self.candidates_))
            return NiceDict({only_candidate: (0, self.first_round_scores_[only_candidate])})

        finalist_a, finalist_b = self.finalists_
        runoff_votes = {candidate: 0 for candidate in self.finalists_}

        for ballot in self.profile_converted_:
            score_a = ballot[finalist_a]
            score_b = ballot[finalist_b]
            if score_a > score_b:
                runoff_votes[finalist_a] += 1
            elif score_b > score_a:
                runoff_votes[finalist_b] += 1

        return NiceDict({
            candidate: (
                runoff_votes[candidate] if candidate in runoff_votes else -1,
                self.first_round_scores_[candidate],
            )
            for candidate in self.candidates_
        })

    def compare_scores(self, one: tuple[int, object], another: tuple[int, object]) -> int:
        if one == another:
            return 0
        if one[0] != another[0]:
            return -1 if one[0] < another[0] else 1
        return -1 if one[1] < another[1] else 1


    