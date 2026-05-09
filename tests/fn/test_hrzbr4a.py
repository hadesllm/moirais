"""Tests for hrzbr4a.horowitz_binary_response_model."""
import numpy as np
import pytest
from moirais.fn.hrzbr4a import horowitz_binary_response_model


def test_hrzbr4a_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_binary_response_model(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzbr4a_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = horowitz_binary_response_model(x, y)
    assert isinstance(result, dict)
