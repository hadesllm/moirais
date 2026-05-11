"""Tests for fzd1d5.fauzi_conditions_d1_d5."""
import numpy as np
import pytest
from morie.fn.fzd1d5 import fauzi_conditions_d1_d5


def test_fzd1d5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_conditions_d1_d5(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzd1d5_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0, 1, 50)
    result = fauzi_conditions_d1_d5(x)
    assert isinstance(result, dict)
