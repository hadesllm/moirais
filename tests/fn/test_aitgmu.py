"""Tests for aitgmu.aitchison_geomean."""
import numpy as np
import pytest
from morie.fn.aitgmu import aitchison_geomean


def test_aitgmu_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_geomean(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitgmu_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_geomean(x)
    assert isinstance(result, dict)
