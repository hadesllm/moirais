"""Tests for morie.fn.odm_p — OTIS demo proportion."""

import pytest
import pandas as pd
from morie.fn.odm_p import otis_demo_proportion
from morie.fn._containers import DescriptiveResult


class TestOtisDemoProportion:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B", "C"]})
        result = otis_demo_proportion(df)
        assert isinstance(result, DescriptiveResult)

    def test_ci_contains_proportion(self):
        df = pd.DataFrame({"group": ["X"] * 30 + ["Y"] * 70})
        result = otis_demo_proportion(df)
        for g, v in result.extra["proportions"].items():
            assert v["ci_lower"] <= v["proportion"] <= v["ci_upper"]
