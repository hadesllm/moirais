"""Tests for moirais.fn.sirsd -- SIRS with demographics."""

import numpy as np
import pytest
from moirais.fn.sirsd import sirs_demographics


class TestSIRSD:
    def test_no_demographics_conservation(self):
        res = sirs_demographics(beta=0.3, gamma=0.1, xi=0.01, mu=0.0, t_max=200)
        total = res.S + res.I + res.R
        np.testing.assert_allclose(total, 1000.0, atol=0.1)

    def test_waning_immunity(self):
        res = sirs_demographics(beta=0.3, gamma=0.1, xi=0.05, mu=0.0, t_max=500)
        S_final = res.S[-1]
        assert S_final > 10

    def test_r0_formula(self):
        res = sirs_demographics(beta=0.5, gamma=0.2, xi=0.01, mu=0.01)
        assert res.R0 == pytest.approx(0.5 / (0.2 + 0.01), rel=1e-6)

    def test_model_label(self):
        res = sirs_demographics(beta=0.3, gamma=0.1, xi=0.01)
        assert res.model == "SIRS-D"

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            sirs_demographics(beta=-0.1, gamma=0.1, xi=0.01)
