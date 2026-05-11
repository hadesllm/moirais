"""Tests for morie.fn.odm_s — OTIS demo standardize."""

import pytest
import numpy as np
from morie.fn.odm_s import otis_demo_standardize
from morie.fn._containers import DescriptiveResult


class TestOtisDemoStandardize:

    def test_returns_descriptive(self):
        obs = np.array([10, 20, 30])
        std = np.array([1000, 2000, 3000])
        result = otis_demo_standardize(obs, std)
        assert isinstance(result, DescriptiveResult)

    def test_equal_rates(self):
        obs = np.array([10, 20])
        std = np.array([100, 200])
        result = otis_demo_standardize(obs, std)
        assert result.extra["standardized_rate"] == pytest.approx(0.1, rel=0.01)
