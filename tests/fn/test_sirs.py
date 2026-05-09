"""Tests for moirais.fn.sirs — SIRS compartmental model."""

import numpy as np
import pytest

from moirais.fn.sirs import sirs_model


class TestSIRSModel:
    def test_returns_sir_result(self):
        res = sirs_model(beta=0.3, gamma=0.1, xi=0.01)
        assert res.model == "SIRS"
        assert res.R is not None
        assert res.R0 == pytest.approx(3.0)

    def test_population_conserved(self):
        res = sirs_model(beta=0.4, gamma=0.2, xi=0.05, n_steps=500)
        total = res.S + res.I + res.R
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_invalid_xi(self):
        with pytest.raises(ValueError):
            sirs_model(beta=0.3, gamma=0.1, xi=-0.01)
