"""Tests for moirais.fn.sird -- SIR compartmental model."""

import numpy as np
import pytest
from moirais.fn.sird import sir_model


class TestSIRModel:
    def test_peak_infection(self):
        """R0=3 should produce a peak infection > initial."""
        res = sir_model(beta=0.3, gamma=0.1, N=1000, I0=1, t_max=200)
        assert res.model == "SIR"
        assert np.max(res.I) > 1.0

    def test_conservation(self):
        """S + I + R = N at all time points."""
        res = sir_model(beta=0.3, gamma=0.1, N=1000, I0=1, t_max=100)
        total = res.S + res.I + res.R
        np.testing.assert_allclose(total, 1000.0, atol=0.1)

    def test_r0_computed(self):
        """R0 = beta/gamma = 3.0."""
        res = sir_model(beta=0.3, gamma=0.1)
        assert res.R0 == pytest.approx(3.0)
