"""Tests for morie.fn.sveir -- SVEIR model."""

import numpy as np
import pytest
from morie.fn.sveir import sveir_model


class TestSVEIR:
    def test_conservation(self):
        res = sveir_model(beta=0.3, sigma=0.2, gamma=0.1, p=0.01)
        total = res.S + res.extra["V"] + res.E + res.I + res.R
        np.testing.assert_allclose(total, 1000.0, atol=0.1)

    def test_vaccination_reduces_peak(self):
        no_vacc = sveir_model(beta=0.5, sigma=0.2, gamma=0.1, p=0.0, t_max=300)
        with_vacc = sveir_model(beta=0.5, sigma=0.2, gamma=0.1, p=0.05, eta=0.9, t_max=300)
        assert np.max(with_vacc.I) < np.max(no_vacc.I)

    def test_perfect_vaccine(self):
        res = sveir_model(beta=0.5, sigma=0.2, gamma=0.1, p=0.1, eta=1.0, t_max=300)
        assert res.extra["V"] is not None
        assert np.max(res.extra["V"]) > 0

    def test_r0(self):
        res = sveir_model(beta=0.6, sigma=0.2, gamma=0.3, p=0.01)
        assert res.R0 == pytest.approx(2.0, rel=1e-6)

    def test_invalid_eta(self):
        with pytest.raises(ValueError):
            sveir_model(beta=0.3, sigma=0.2, gamma=0.1, p=0.01, eta=1.5)

    def test_model_label(self):
        res = sveir_model(beta=0.3, sigma=0.2, gamma=0.1, p=0.01)
        assert res.model == "SVEIR"
