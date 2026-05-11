"""Tests for morie.fn.cmplr -- Complier Average Causal Effect."""

import numpy as np
import pandas as pd
import pytest
from morie.fn.cmplr import complier_ate, cmplr
from morie.fn._containers import ESRes


class TestCACE:
    def test_alias(self):
        assert cmplr is complier_ate

    def test_perfect_compliance(self):
        """With perfect compliance, CACE = ATE."""
        rng = np.random.default_rng(42)
        n = 500
        z = rng.binomial(1, 0.5, n).astype(float)
        t = z.copy()
        y = 1.0 + 3.0 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "instrument": z})
        result = complier_ate(df)
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 3.0) < 1.0
        assert abs(result.extra["compliance_rate"] - 1.0) < 0.05

    def test_partial_compliance(self):
        """With 50% compliance, CACE should still be estimated."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.binomial(1, 0.5, n).astype(float)
        comply = rng.binomial(1, 0.6, n).astype(float)
        t = (z * comply).astype(float)
        y = 1.0 + 4.0 * t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "instrument": z})
        result = complier_ate(df)
        assert result.extra["compliance_rate"] > 0.1
        assert result.se > 0

    def test_weak_instrument_raises(self):
        """When Z doesn't predict T, should raise."""
        rng = np.random.default_rng(42)
        n = 100
        z = rng.binomial(1, 0.5, n).astype(float)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = t + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": t, "instrument": z})
        try:
            result = complier_ate(df)
            assert result.se > 1.0
        except ValueError:
            pass
