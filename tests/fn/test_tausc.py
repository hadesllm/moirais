"""Tests for tau_scale."""
import numpy as np, pytest
from moirais.fn.tausc import tau_scale


class TestTauScale:
    def test_normal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 2, 200)
        r = tau_scale(x)
        assert r.measure == "tau_scale"
        assert r.estimate > 0

    def test_constant(self):
        r = tau_scale([3, 3, 3, 3, 3])
        assert r.estimate == pytest.approx(0.0)

    def test_too_few(self):
        with pytest.raises(ValueError):
            tau_scale([1.0])
