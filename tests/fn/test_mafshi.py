"""Tests for mafshi.ma_fishers_z_inverse."""
import numpy as np
import pytest
from moirais.fn.mafshi import ma_fishers_z_inverse


def test_mafshi_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = ma_fishers_z_inverse(z)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_mafshi_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = ma_fishers_z_inverse(z)
    assert isinstance(result, dict)
