"""Tests for moirais.fn.garch — GARCH(1,1)."""
import numpy as np
import pytest
from moirais.fn.garch import garch_fit


class TestGARCH:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = rng.standard_normal(200) * 0.01
        res = garch_fit(r)
        assert res.extra["alpha"] >= 0
        assert res.extra["beta"] >= 0

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            garch_fit(np.ones(5))
