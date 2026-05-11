"""Tests for latent_growth."""
import numpy as np, pytest
from morie.fn.lcgm import latent_growth

class TestLCGM:
    def test_linear_growth(self):
        Y = np.array([[1,2,3,4],[2,4,6,8],[0,1,2,3]], dtype=float)
        r = latent_growth(Y)
        assert r.extra["mean_slope"] == pytest.approx(1.0, abs=0.5)

    def test_no_growth(self):
        Y = np.ones((5, 4))
        r = latent_growth(Y)
        assert r.extra["mean_slope"] == pytest.approx(0.0, abs=1e-6)
