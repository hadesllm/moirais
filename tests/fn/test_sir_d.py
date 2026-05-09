"""Tests for moirais.fn.sir_d — SIR with demography."""

import numpy as np
import pytest

from moirais.fn.sir_d import sir_demography


class TestSIRDemography:
    def test_returns_sir_result(self):
        res = sir_demography(beta=0.3, gamma=0.1, mu=0.01)
        assert res.model == "SIR-D"
        assert res.R0 == pytest.approx(0.3 / 0.11, rel=1e-4)

    def test_endemic_equilibrium(self):
        res = sir_demography(beta=0.5, gamma=0.1, mu=0.02, t_max=2000, n_steps=5000)
        assert res.I[-1] > 0.001

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            sir_demography(beta=0, gamma=0.1)
