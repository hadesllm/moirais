"""Tests for moirais.fn.tpsnb — neighbourhood."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.tpsnb import tps_neighborhood
from moirais.fn._containers import DescriptiveResult


class TestNeighbourhood:
    def test_basic(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"neighbourhood": rng.choice(["A", "B", "C"], 100), "crime_type": "Theft"})
        r = tps_neighborhood(df)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_neighbourhoods"] == 3

    def test_top5(self):
        df = pd.DataFrame({"neighbourhood": ["X"] * 50 + ["Y"] * 10})
        r = tps_neighborhood(df)
        assert "X" in r.extra["top5"]
