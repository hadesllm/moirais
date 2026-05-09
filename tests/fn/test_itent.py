"""Tests for moirais.fn.itent — item entropy."""

import numpy as np
import pandas as pd
from moirais.fn.itent import item_entropy


class TestItemEntropy:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = item_entropy(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "entropy" in result.columns
        assert "relative_entropy" in result.columns

    def test_constant_item_zero_entropy(self):
        data = np.full((50, 2), 3.0)
        result = item_entropy(data)
        assert abs(result["entropy"].iloc[0]) < 1e-10

    def test_uniform_max_entropy(self):
        # 5 options, each chosen equally
        rng = np.random.default_rng(42)
        data = np.tile(np.arange(1, 6), 100).reshape(-1, 1)
        rng.shuffle(data)
        result = item_entropy(data)
        assert abs(result["relative_entropy"].iloc[0] - 1.0) < 0.01

    def test_entropy_non_negative(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_entropy(mapq_df[items])
        assert (result["entropy"] >= 0).all()

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(100, 4))
        result = item_entropy(data)
        assert len(result) == 4
