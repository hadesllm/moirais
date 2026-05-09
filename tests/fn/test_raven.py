"""Tests for moirais.fn.raven -- Raven's Progressive Matrices scoring."""

import numpy as np
from moirais.fn.raven import raven_score, raven
from moirais.fn._containers import DescriptiveResult


class TestRaven:
    def test_alias(self):
        assert raven is raven_score

    def test_perfect_score(self):
        correct = list(range(1, 31))
        result = raven_score(correct, correct, n_sets=5)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 30
        assert result.extra["pct_correct"] == 100.0

    def test_half_score(self):
        correct = [1] * 30
        responses = [1] * 15 + [2] * 15
        result = raven_score(responses, correct)
        assert result.value == 15
