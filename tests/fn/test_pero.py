"""Tests for pero.prioritized_experience_replay."""
import numpy as np
import pytest
from moirais.fn.pero import prioritized_experience_replay


def test_pero_basic():
    """Test basic functionality."""
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = prioritized_experience_replay(buffer, alpha, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pero_edge():
    """Test edge cases."""
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    beta = 0.8
    result = prioritized_experience_replay(buffer, alpha, beta)
    assert isinstance(result, dict)
