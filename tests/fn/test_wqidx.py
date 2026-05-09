"""Tests for moirais.fn.wqidx -- water quality index."""

import pytest
from moirais.fn.wqidx import water_quality_index


class TestWQI:
    def test_good(self):
        res = water_quality_index({"DO": 80, "pH": 75, "turbidity": 70})
        assert res.extra["quality"] == "good"

    def test_weighted(self):
        res = water_quality_index({"DO": 90, "pH": 50}, weights={"DO": 3, "pH": 1})
        assert res.estimate == pytest.approx(80.0)

    def test_empty(self):
        with pytest.raises(ValueError):
            water_quality_index({})
