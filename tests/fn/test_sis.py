"""Tests for morie.fn.sis — SIS compartmental model."""

import numpy as np
import pytest

from morie.fn.sis import sis_model


class TestSISModel:
    def test_returns_sir_result(self):
        res = sis_model(beta=0.3, gamma=0.1)
        assert res.model == "SIS"
        assert res.R is None
        assert res.R0 == pytest.approx(3.0)

    def test_population_conserved(self):
        res = sis_model(beta=0.4, gamma=0.2, n_steps=500)
        total = res.S + res.I
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            sis_model(beta=-0.1, gamma=0.1)
