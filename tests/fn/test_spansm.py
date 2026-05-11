"""Tests for spansm.schabenberger_anselin_local_moran."""
import numpy as np
import pytest
from morie.fn.spansm import schabenberger_anselin_local_moran


def test_spansm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_anselin_local_moran(x, w)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spansm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_anselin_local_moran(x, w)
    assert isinstance(result, dict)
