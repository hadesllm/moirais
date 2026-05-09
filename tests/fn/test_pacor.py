"""Tests for pacor.predictive_ability_pearson."""
import numpy as np
import pytest
from moirais.fn.pacor import predictive_ability_pearson


def test_pacor_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = predictive_ability_pearson(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_pacor_edge():
    """Test edge cases."""
    result = predictive_ability_pearson(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
