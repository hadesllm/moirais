"""Tests for moirais.fn.bycpi -- BayesCpi genomic prediction."""

import numpy as np
import pytest
from moirais.fn.bycpi import bycpi


def _sim(n=60, p=10, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    beta_true = np.zeros(p)
    beta_true[:3] = [1.0, -0.5, 0.8]
    y = Z @ beta_true + rng.standard_normal(n) * 0.5
    return y, Z


class TestBycpi:
    def test_returns_genomics_result(self):
        y, Z = _sim()
        res = bycpi(y, Z, n_iter=200, burn_in=50)
        assert res.name == "BayesCpi"

    def test_pi_estimated(self):
        y, Z = _sim()
        res = bycpi(y, Z, n_iter=200, burn_in=50)
        pi = res.extra["pi_est"]
        assert 0 < pi < 1

    def test_effects_length(self):
        y, Z = _sim()
        res = bycpi(y, Z, n_iter=200, burn_in=50)
        assert len(res.extra["effects"]) == Z.shape[1]

    def test_dimension_mismatch(self):
        y, Z = _sim()
        with pytest.raises(ValueError):
            bycpi(y[:5], Z)
