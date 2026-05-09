"""Tests for moirais.fn.ortho -- Neyman-orthogonal score."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.ortho import orthogonal_score, ortho
from moirais.fn._containers import ESRes


class TestOrthogonalScore:
    def test_alias(self):
        assert ortho is orthogonal_score

    def test_known_theta(self):
        """With perfect nuisance, should recover true theta."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        t = rng.binomial(1, 0.5, n).astype(float)
        y = 2.0 * t + 0.5 * x + rng.normal(0, 0.3, n)
        m_hat = 0.5 * x
        e_hat = np.full(n, 0.5)
        df = pd.DataFrame({"outcome": y, "treatment": t, "m_hat": m_hat, "e_hat": e_hat})
        result = orthogonal_score(df)
        assert isinstance(result, ESRes)
        assert abs(result.estimate - 2.0) < 0.5

    def test_degenerate_raises(self):
        df = pd.DataFrame({
            "outcome": [1.0, 2.0],
            "treatment": [0.5, 0.5],
            "m_hat": [1.0, 2.0],
            "e_hat": [0.5, 0.5],
        })
        with pytest.raises(ValueError, match="zero"):
            orthogonal_score(df)
