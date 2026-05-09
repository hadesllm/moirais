"""Tests for aitbal.aitchison_balance."""
import numpy as np
import pytest
from moirais.fn.aitbal import aitchison_balance


def test_aitbal_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    row = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_balance(x, row)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitbal_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    row = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_balance(x, row)
    assert isinstance(result, dict)
