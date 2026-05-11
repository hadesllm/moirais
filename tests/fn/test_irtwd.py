"""Tests for morie.fn.irtwd — Wright map."""

import numpy as np
import pandas as pd
from morie.fn.irtwd import irt_wright_map


class TestIrtWrightMap:
    def test_returns_dict(self, rng):
        params = {f"i{j}": {"b": float(rng.standard_normal())} for j in range(10)}
        theta = rng.standard_normal(100)
        result = irt_wright_map(params, theta)
        assert isinstance(result, dict)
        assert "items" in result
        assert "persons" in result
        assert "alignment" in result

    def test_items_dataframe(self, rng):
        params = {"i1": {"b": -1.0}, "i2": {"b": 0.0}, "i3": {"b": 1.0}}
        theta = rng.standard_normal(50)
        result = irt_wright_map(params, theta)
        assert isinstance(result["items"], pd.DataFrame)
        assert len(result["items"]) == 3

    def test_person_stats(self, rng):
        params = {"i1": {"b": 0.0}}
        theta = rng.standard_normal(200)
        result = irt_wright_map(params, theta)
        assert result["persons"]["n"] == 200
        assert abs(result["persons"]["mean"]) < 0.3  # approx zero

    def test_alignment_range(self, rng):
        params = {"i1": {"b": -2.0}, "i2": {"b": 2.0}}
        theta = rng.standard_normal(500)
        result = irt_wright_map(params, theta)
        assert 0 <= result["alignment"]["pct_persons_in_range"] <= 1
