"""Tests for morie.fn.did -- Difference-in-differences."""

import numpy as np
import pandas as pd
from morie.fn.did import diff_in_diff, did
from morie.fn._containers import ESRes


class TestDiD:
    def test_alias(self):
        assert did is diff_in_diff

    def test_known_effect(self):
        """2x2 DiD with true effect = 3."""
        rng = np.random.default_rng(42)
        n_per_cell = 200
        rows = []
        for treat in [0, 1]:
            for post_val in [0, 1]:
                base = 10 + 2 * treat + 1 * post_val
                effect = 3 * treat * post_val  # DiD effect = 3
                for _ in range(n_per_cell):
                    rows.append({
                        "outcome": base + effect + rng.normal(0, 1),
                        "treatment": treat,
                        "post": post_val,
                    })
        df = pd.DataFrame(rows)
        result = diff_in_diff(df)
        assert isinstance(result, ESRes)
        assert result.measure == "DiD"
        assert abs(result.estimate - 3.0) < 0.5
        assert result.ci_lower < 3.0 < result.ci_upper

    def test_zero_effect(self):
        """Parallel trends, no treatment effect."""
        rng = np.random.default_rng(42)
        rows = []
        for treat in [0, 1]:
            for post_val in [0, 1]:
                for _ in range(100):
                    rows.append({
                        "outcome": 5 + 1 * post_val + rng.normal(0, 1),
                        "treatment": treat,
                        "post": post_val,
                    })
        df = pd.DataFrame(rows)
        result = diff_in_diff(df)
        assert abs(result.estimate) < 1.0
