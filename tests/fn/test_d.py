"""Tests for morie.fn.d -- Cohen's d effect size."""

import numpy as np
import pytest
from morie.fn.d import cohens_d


class TestCohensD:
    def test_known_large_effect(self):
        """Well-separated groups should yield large d."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [6.0, 7.0, 8.0, 9.0, 10.0]
        d = cohens_d(x, y)
        assert isinstance(d, float)
        assert abs(d) > 1.5  # very large separation

    def test_identical_groups_zero(self):
        """Identical groups should yield d = 0."""
        x = [1.0, 2.0, 3.0]
        d = cohens_d(x, x)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_sign_direction(self):
        """d should be positive when x1 > x2."""
        x1 = [10.0, 11.0, 12.0]
        x2 = [1.0, 2.0, 3.0]
        d = cohens_d(x1, x2)
        assert d > 0
