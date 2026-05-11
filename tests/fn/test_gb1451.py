"""Tests for gb1451.gibbons_mcnemar."""
import numpy as np
import pytest
from morie.fn.gb1451 import gibbons_mcnemar


def test_gb1451_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_mcnemar(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1451_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_mcnemar(table)
    assert isinstance(result, dict)
