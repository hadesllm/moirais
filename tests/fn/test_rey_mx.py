"""Tests for morie.fn.rey_mx — mixed-effects model."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.rey_mx import rey_mx


class TestReyMx:
    """Tests for rey_mx()."""

    def test_basic_random_intercept(self):
        """Recovers fixed effects with random intercepts."""
        rng = np.random.default_rng(42)
        groups = np.repeat(np.arange(20), 10)
        b = rng.normal(0, 2, 20)
        x = rng.standard_normal(200)
        y = 3.0 + 1.5 * x + b[groups] + rng.normal(0, 0.5, 200)
        df = pd.DataFrame({"y": y, "x": x, "group": groups})
        result = rey_mx(df, y="y", x_fixed="x", group_col="group")
        assert "fixed_effects" in result
        assert result["n_groups"] == 20
        # Fixed slope should be near 1.5
        assert abs(result["fixed_effects"]["x"] - 1.5) < 0.5

    def test_variance_components_positive(self):
        """Variance components are positive."""
        rng = np.random.default_rng(7)
        groups = np.repeat(np.arange(10), 15)
        x = rng.standard_normal(150)
        y = 1.0 + x + rng.normal(0, 1, 10)[groups] + rng.normal(0, 0.5, 150)
        df = pd.DataFrame({"y": y, "x": x, "g": groups})
        result = rey_mx(df, y="y", x_fixed="x", group_col="g")
        assert result["variance_components"]["sigma2_b"] > 0
        assert result["variance_components"]["sigma2_e"] > 0

    def test_raises_missing_column(self):
        """Missing column raises ValueError."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="not found"):
            rey_mx(df, y="y", x_fixed="a", group_col="b")
