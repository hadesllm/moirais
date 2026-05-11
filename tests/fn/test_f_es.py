"""Tests for morie.fn.f_es -- Cohen's f from eta-squared."""

import math
import pytest
from morie.fn.f_es import cohens_f
from morie.fn._containers import ESRes


class TestCohensF:
    def test_known_value(self):
        """f = sqrt(eta2 / (1 - eta2))."""
        eta2 = 0.25
        expected = math.sqrt(0.25 / 0.75)  # sqrt(1/3) ~ 0.5774
        result = cohens_f(eta2)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(expected, abs=1e-4)

    def test_zero_eta2(self):
        """eta2 = 0 gives f = 0."""
        result = cohens_f(0.0)
        assert result.estimate == pytest.approx(0.0)

    def test_small_medium_large(self):
        """Conventional benchmarks: f=0.10 small, 0.25 medium, 0.40 large."""
        small = cohens_f(0.01)
        medium = cohens_f(0.06)
        large = cohens_f(0.14)
        assert small.estimate < medium.estimate < large.estimate
