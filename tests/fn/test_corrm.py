"""Tests for correlation_matrix."""
import numpy as np, pandas as pd, pytest
from moirais.fn.corrm import correlation_matrix

class TestCorrm:
    def test_basic(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(0, 1, (30, 3)), columns=["a", "b", "c"])
        r = correlation_matrix(df)
        assert r.value.shape == (3, 3)

    def test_perfect(self):
        df = pd.DataFrame({"x": [1,2,3,4,5], "y": [2,4,6,8,10]})
        r = correlation_matrix(df)
        assert r.value.loc["x", "y"] == pytest.approx(1.0)
