"""Tests for moirais.fn.w -- Cohen's w for chi-squared."""

import pytest
from moirais.fn.w import cohens_w
from moirais.fn._containers import ESRes


class TestCohensW:
    def test_uniform_observed_near_zero(self):
        """Uniform observed matching expected gives w near 0."""
        result = cohens_w([25, 25, 25, 25])
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(0.0, abs=1e-10)

    def test_skewed_large_w(self):
        """Highly skewed frequencies yield large w."""
        result = cohens_w([90, 5, 3, 2])
        assert result.estimate > 0.5

    def test_with_explicit_expected(self):
        """Custom expected frequencies."""
        result = cohens_w([40, 60], expected=[50, 50])
        assert result.estimate > 0
