"""Tests for moirais.fn.scpmp — percent of maximum possible score."""

import numpy as np
import pandas as pd
from moirais.fn.scpmp import score_pmp


class TestScorePmp:
    def test_returns_series(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        result = score_pmp(mapq_df, items=items, min_val=1, max_val=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(mapq_df)

    def test_range_0_100(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        result = score_pmp(mapq_df[items], min_val=1, max_val=5)
        assert (result >= -0.01).all()
        assert (result <= 100.01).all()

    def test_known_values(self):
        df = pd.DataFrame({"a": [1, 3, 5], "b": [1, 3, 5]})
        result = score_pmp(df, min_val=1, max_val=5)
        # min sum=2, max sum=10, range=8
        # [2,6,10] -> [0, 50, 100]
        assert np.allclose(result.values, [0.0, 50.0, 100.0])

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(50, 5))
        result = score_pmp(data, min_val=1, max_val=5)
        assert len(result) == 50
