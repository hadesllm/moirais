"""Tests for estnd.estimand_framework."""
import numpy as np
import pytest
from morie.fn.estnd import estimand_framework


def test_estnd_basic():
    """Test basic functionality."""
    estimand_type = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = estimand_framework(estimand_type, data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_estnd_edge():
    """Test edge cases."""
    estimand_type = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = estimand_framework(estimand_type, data)
    assert isinstance(result, dict)
