"""Tests for morie.fn.itdif — item difficulty."""

import numpy as np
import pandas as pd
from morie.fn.itdif import item_difficulty


class TestItemDifficulty:
    def test_returns_dataframe(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_difficulty(mapq_df[items])
        assert isinstance(result, pd.DataFrame)
        assert "item" in result.columns
        assert "difficulty" in result.columns
        assert len(result) == len(items)

    def test_difficulty_range(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = item_difficulty(mapq_df[items])
        assert (result["difficulty"] >= 0).all()
        assert (result["difficulty"] <= 1).all()

    def test_ndarray_input(self, rng):
        data = rng.integers(1, 6, size=(100, 5))
        result = item_difficulty(data)
        assert len(result) == 5
        assert result["item"].tolist() == [f"i{i}" for i in range(5)]

    def test_perfect_scores(self):
        data = np.ones((50, 3))
        result = item_difficulty(data)
        assert np.allclose(result["difficulty"].values, 1.0)

    def test_binary_items(self, mapq_binary_df):
        result = item_difficulty(mapq_binary_df)
        assert len(result) == 10
        # Difficulty = mean for binary (max=1)
        for i, row in result.iterrows():
            col = mapq_binary_df.iloc[:, i]
            expected = col.mean()
            assert abs(row["difficulty"] - expected) < 1e-10
