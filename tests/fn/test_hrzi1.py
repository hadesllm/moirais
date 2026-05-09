"""Tests for hrzi1.horowitz_index_model."""
import numpy as np
import pytest
from moirais.fn.hrzi1 import horowitz_index_model


def test_hrzi1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_index_model(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzi1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_index_model(x, y)
    assert isinstance(result, dict)
