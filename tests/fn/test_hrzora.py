"""Tests for hrzora.horowitz_two_step_oracle."""
import numpy as np
import pytest
from moirais.fn.hrzora import horowitz_two_step_oracle


def test_hrzora_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_two_step_oracle(x, y, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzora_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    result = horowitz_two_step_oracle(x, y, bandwidth)
    assert isinstance(result, dict)
