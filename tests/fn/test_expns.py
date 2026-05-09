"""Tests for moirais.fn.expns — exponential smoothing."""
import numpy as np
import pytest
from moirais.fn.expns import exponential_smooth


class TestExponentialSmooth:
    def test_basic(self):
        y = np.random.default_rng(42).standard_normal(50)
        res = exponential_smooth(y, alpha=0.3)
        assert len(res.extra["smoothed"]) == 50

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            exponential_smooth(np.ones(10), alpha=0.0)
