"""Tests for moirais.fn.scmn — mean score."""

import numpy as np
import pandas as pd
from moirais.fn.scmn import score_mean


class TestScoreMean:
    def test_returns_series(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_mean(mapq_df, items=items)
        assert isinstance(result, pd.Series)
        assert len(result) == len(mapq_df)

    def test_mean_correct(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]})
        result = score_mean(df)
        assert np.allclose(result.values, [2.0, 3.0, 4.0])

    def test_range(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = score_mean(mapq_df[items])
        assert (result >= 1).all()
        assert (result <= 5).all()

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(50, 5))
        result = score_mean(data)
        assert len(result) == 50
