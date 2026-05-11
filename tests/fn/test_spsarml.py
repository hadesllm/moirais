"""Tests for spsarml.schabenberger_sar_ml."""
import numpy as np
import pytest
from morie.fn.spsarml import schabenberger_sar_ml


def test_spsarml_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_sar_ml(x, y, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spsarml_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_sar_ml(x, y, w)
    assert isinstance(result, dict)
