"""Tests for moirais.fn.relps -- Relapse vs reinfection (Pv3Rs)."""

import numpy as np
import pytest
from moirais.fn.relps import relps


class TestRelps:
    def test_identical_alleles_relapse(self):
        a0 = np.array([1, 2, 3, 4, 5])
        a1 = np.array([1, 2, 3, 4, 5])
        pf = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        res = relps(a0, a1, allele_freqs=pf)
        assert res.statistic > 0.9

    def test_different_alleles_reinfection(self):
        a0 = np.array([1, 2, 3])
        a1 = np.array([6, 7, 8])
        pf = np.array([0.5, 0.5, 0.5])
        res = relps(a0, a1, allele_freqs=pf)
        assert res.statistic < 0.5

    def test_p_relapse_plus_reinfection(self):
        a0 = np.array([1, 2, 3])
        a1 = np.array([1, 2, 8])
        pf = np.array([0.1, 0.1, 0.1])
        res = relps(a0, a1, allele_freqs=pf)
        assert abs(res.statistic + res.p_value - 1.0) < 1e-10

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            relps(np.array([1, 2]), np.array([1]))
