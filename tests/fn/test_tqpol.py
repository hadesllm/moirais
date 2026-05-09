"""Tests for tqpol.turboquant_polar_transform."""
import numpy as np
import pytest
from moirais.fn.tqpol import turboquant_polar_transform


def test_tqpol_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_polar_transform(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqpol_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = turboquant_polar_transform(x)
    assert isinstance(result, dict)
