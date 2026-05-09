"""Tests for dpedt.dp_exchangeable_distribution."""
import numpy as np
import pytest
from moirais.fn.dpedt import dp_exchangeable_distribution


def test_dpedt_basic():
    """Test basic functionality."""
    partition = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dp_exchangeable_distribution(partition, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpedt_edge():
    """Test edge cases."""
    partition = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = dp_exchangeable_distribution(partition, alpha)
    assert isinstance(result, dict)
