"""Tests for moirais.fn.lambd — genomic inflation factor."""
import numpy as np
import pytest
from moirais.fn.lambd import genomic_inflation


class TestGenomicInflation:
    def test_no_inflation(self):
        from scipy import stats
        chi2 = stats.chi2.rvs(1, size=1000, random_state=42)
        res = genomic_inflation(chi2_stats=chi2)
        assert res.estimate == pytest.approx(1.0, abs=0.15)

    def test_from_pvalues(self):
        pv = np.random.default_rng(42).uniform(0.001, 1.0, 500)
        res = genomic_inflation(p_values=pv)
        assert res.estimate > 0

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            genomic_inflation()
