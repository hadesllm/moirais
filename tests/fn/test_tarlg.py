"""Tests for morie.fn.tarlg -- Threshold AR."""
import numpy as np
import pytest
from morie.fn.tarlg import tar_fit


class TestTAR:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal(200)
        res = tar_fit(y, p=1)
        assert "phi_low" in res.extra
        assert "phi_high" in res.extra

    def test_short_raises(self):
        with pytest.raises(ValueError):
            tar_fit(np.ones(5), p=1)

    def test_cheatsheet(self):
        from morie.fn.tarlg import cheatsheet
        assert isinstance(cheatsheet(), str)
