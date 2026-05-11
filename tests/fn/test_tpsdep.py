"""Tests for morie.fn.tpsdep — deployment."""

import pytest
import pandas as pd
from morie.fn.tpsdep import tps_deployment
from morie.fn._containers import DescriptiveResult


class TestDeployment:
    def test_basic(self):
        df = pd.DataFrame({"division": ["D11", "D11", "D52", "D52"], "officer_count": [100, 50, 80, 70]})
        r = tps_deployment(df)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(300.0)

    def test_proportions(self):
        df = pd.DataFrame({"division": ["A", "B"], "officer_count": [50, 50]})
        r = tps_deployment(df)
        assert r.extra["proportions"]["A"] == pytest.approx(0.5)

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            tps_deployment(df)
