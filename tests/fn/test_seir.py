"""Tests for moirais.fn.seir -- SEIR compartmental model."""

import numpy as np
import pytest
from moirais.fn.seir import seir_model


class TestSEIRModel:
    def test_exposed_populated(self):
        """E compartment should become populated during epidemic."""
        res = seir_model(beta=0.3, sigma=0.2, gamma=0.1, N=1000, I0=1, t_max=200)
        assert res.model == "SEIR"
        assert res.E is not None
        assert np.max(res.E) > 0.1

    def test_conservation(self):
        """S + E + I + R = N at all time points."""
        res = seir_model(beta=0.3, sigma=0.2, gamma=0.1, N=1000, I0=1, t_max=100)
        total = res.S + res.E + res.I + res.R
        np.testing.assert_allclose(total, 1000.0, atol=0.1)

    def test_r0(self):
        """R0 = beta/gamma."""
        res = seir_model(beta=0.5, sigma=0.2, gamma=0.25)
        assert res.R0 == pytest.approx(2.0)
