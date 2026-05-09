"""Tests for moirais.fn.bunck -- Bunching estimator."""

import numpy as np
import pandas as pd
from moirais.fn.bunck import bunching, bunck
from moirais.fn._containers import ESRes


class TestBunching:
    def test_alias(self):
        assert bunck is bunching

    def test_detects_bunching(self):
        """With extra mass at the cutoff, excess should be positive."""
        rng = np.random.default_rng(42)
        n_base = 500
        r_base = rng.normal(0, 2, n_base)
        n_bunch = 100
        r_bunch = rng.normal(0, 0.1, n_bunch)
        r = np.concatenate([r_base, r_bunch])
        df = pd.DataFrame({"running": r})
        result = bunching(df, cutoff=0.0)
        assert isinstance(result, ESRes)
        assert result.estimate > 0

    def test_no_bunching(self):
        """Uniform data should show little excess."""
        rng = np.random.default_rng(42)
        r = rng.uniform(-5, 5, 1000)
        df = pd.DataFrame({"running": r})
        result = bunching(df, cutoff=0.0, bin_width=0.5)
        assert abs(result.estimate) < 50

    def test_se_positive(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 1, 500)
        df = pd.DataFrame({"running": r})
        result = bunching(df)
        assert result.se > 0
