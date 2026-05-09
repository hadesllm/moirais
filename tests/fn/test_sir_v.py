"""Tests for moirais.fn.sir_v — SIR with vaccination."""

import numpy as np
import pytest

from moirais.fn.sir_v import sir_vaccination


class TestSIRVaccination:
    def test_vaccination_reduces_peak(self):
        no_vax = sir_vaccination(beta=0.3, gamma=0.1, v=0.0)
        with_vax = sir_vaccination(beta=0.3, gamma=0.1, v=0.02)
        assert np.max(with_vax.I) < np.max(no_vax.I)

    def test_population_conserved(self):
        res = sir_vaccination(beta=0.3, gamma=0.1, v=0.01)
        total = res.S + res.I + res.R
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_negative_vax(self):
        with pytest.raises(ValueError):
            sir_vaccination(v=-0.01)
