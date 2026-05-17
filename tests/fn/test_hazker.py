"""Tests for morie.fn.hazker -- Kernel-smoothed hazard rate."""

import numpy as np
import pandas as pd
from morie.fn.hazker import hazard_kernel, hazker
from morie.fn._containers import DescriptiveResult


class TestHazker:
    def test_alias(self):
        assert hazker is hazard_kernel

    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "time": rng.exponential(5, 50),
            "event": rng.binomial(1, 0.8, 50),
        })
        result = hazard_kernel(df)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["hazard"]) == 100

    def test_all_events(self):
        df = pd.DataFrame({"time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "event": [1]*10})
        result = hazard_kernel(df, n_grid=20)
        assert result.extra["n_events"] == 10
