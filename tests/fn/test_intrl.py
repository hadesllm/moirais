"""Tests for moirais.fn.intrl -- Interrupted time series."""

import pytest
import numpy as np
from moirais.fn.intrl import interrupted_time_series


class TestITS:
    def test_level_change(self):
        pre = [10 + np.random.default_rng(42).normal() for _ in range(20)]
        post = [15 + np.random.default_rng(43).normal() for _ in range(20)]
        y = pre + post
        res = interrupted_time_series(y, intervention_point=20)
        assert res.measure == "ITS"
        assert res.extra["level_change"] > 0

    def test_coefficients(self):
        y = list(range(40))
        res = interrupted_time_series(y, intervention_point=20)
        assert "pre_trend" in res.extra
        assert "slope_change" in res.extra

    def test_invalid(self):
        with pytest.raises(ValueError):
            interrupted_time_series([1, 2, 3], intervention_point=0)
