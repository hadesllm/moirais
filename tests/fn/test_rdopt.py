"""Tests for morie.fn.rdopt -- RD optimal bandwidth."""

import numpy as np
import pandas as pd
from morie.fn.rdopt import rd_bandwidth, rdopt


class TestRDBandwidth:
    def test_alias(self):
        assert rdopt is rd_bandwidth

    def test_ik_returns_positive(self):
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-3, 3, n)
        y = 2.0 * (r >= 0) + 0.5 * r + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = rd_bandwidth(df)
        assert result["h_opt"] > 0
        assert result["h_pilot"] > 0
        assert result["method"] == "ik"

    def test_cct_method(self):
        rng = np.random.default_rng(42)
        n = 500
        r = rng.uniform(-3, 3, n)
        y = 3.0 * (r >= 0) + rng.normal(0, 1, n)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = rd_bandwidth(df, method="cct")
        assert result["h_opt"] > 0
        assert result["method"] == "cct"

    def test_n_left_right(self):
        rng = np.random.default_rng(42)
        r = rng.uniform(-2, 2, 400)
        y = (r >= 0).astype(float) + rng.normal(0, 1, 400)
        df = pd.DataFrame({"outcome": y, "running": r})
        result = rd_bandwidth(df)
        assert result["n_left"] > 0
        assert result["n_right"] > 0
        assert result["n_left"] + result["n_right"] == 400
