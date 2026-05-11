"""Tests for morie.fn.rskgp — risk group profile."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.rskgp import risk_group_profile
from morie.fn._containers import DescriptiveResult


class TestRiskGroupProfile:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"risk_level": ["Low", "Med", "High", "Low", "Med", "High"],
                           "age": [25, 30, 35, 28, 40, 45]})
        result = risk_group_profile(df)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_groups"] == 3

    def test_custom_cols(self):
        df = pd.DataFrame({"risk_level": [1, 2, 1, 2], "x": [10, 20, 15, 25], "y": [1, 0, 1, 0]})
        result = risk_group_profile(df, profile_cols=["x"])
        assert "profiles" in result.extra
