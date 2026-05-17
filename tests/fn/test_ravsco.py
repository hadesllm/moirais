"""Tests for morie.fn.ravsco -- Raven's Progressive Matrices scoring."""

import numpy as np
from morie.fn.ravsco import raven_score, ravsco
from morie.fn._containers import DescriptiveResult


class TestRavsco:
    def test_alias(self):
        assert ravsco is raven_score

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
