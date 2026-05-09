"""Tests for vcont -- content validity ratio."""
import numpy as np
from moirais.fn.vcont import content_validity_ratio
from moirais.fn._containers import ESRes


class TestContentValidity:
    def test_all_essential(self):
        result = content_validity_ratio(10, 10)
        assert isinstance(result, ESRes)
        assert result.estimate == 1.0

    def test_half_essential(self):
        result = content_validity_ratio(5, 10)
        assert result.estimate == 0.0

    def test_array_items(self):
        result = content_validity_ratio(np.array([8, 9, 10]), 10)
        assert result.extra["n_items"] == 3
