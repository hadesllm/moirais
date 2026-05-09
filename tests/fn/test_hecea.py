"""Tests for moirais.fn.hecea -- CE plane."""

import numpy as np
from moirais.fn.hecea import cost_effectiveness_plane


class TestCEPlane:
    def test_basic(self):
        rng = np.random.default_rng(42)
        res = cost_effectiveness_plane(
            cost_diffs=rng.normal(100, 50, 1000),
            effect_diffs=rng.normal(0.5, 0.3, 1000),
        )
        assert res.name == "ce_plane"
        total = sum(res.value.values())
        assert abs(total - 100.0) < 0.1

    def test_mismatch(self):
        import pytest
        with pytest.raises(ValueError):
            cost_effectiveness_plane([1, 2], [1])
