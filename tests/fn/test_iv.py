"""Tests for moirais.fn.iv -- Instrumental variables (2SLS)."""

import numpy as np
import pandas as pd
from moirais.fn.iv import iv_2sls, iv
from moirais.fn._containers import RegressionResult


class TestIV:
    def test_alias(self):
        assert iv is iv_2sls

    def test_recovers_effect(self):
        """Instrument Z -> D -> Y with true effect = 4, confounder U."""
        rng = np.random.default_rng(42)
        n = 2000
        z = rng.binomial(1, 0.5, n).astype(float)
        u = rng.normal(0, 1, n)  # unobserved confounder
        d = (0.5 * z + 0.5 * u + rng.normal(0, 0.3, n) > 0.5).astype(float)
        y = 4 * d + 2 * u + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": d, "instrument": z})
        result = iv_2sls(df)
        assert isinstance(result, RegressionResult)
        assert result.method == "2SLS"
        # 2SLS should get closer to 4 than OLS would
        assert abs(result.coefficients["treatment"] - 4.0) < 3.0
        assert result.extra["first_stage_F"] > 5

    def test_first_stage_f(self):
        """Strong instrument should have high first-stage F."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.normal(0, 1, n)
        d = 0.8 * z + rng.normal(0, 0.3, n)
        y = 2 * d + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": d, "instrument": z})
        result = iv_2sls(df)
        assert result.extra["first_stage_F"] > 10

    def test_with_covariates(self):
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        z = rng.binomial(1, 0.5, n).astype(float)
        d = (0.5 * z + 0.3 * x1 + rng.normal(0, 0.5, n) > 0.3).astype(float)
        y = 3 * d + x1 + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "treatment": d, "instrument": z, "x1": x1})
        result = iv_2sls(df, x="x1")
        assert "x1" in result.coefficients
