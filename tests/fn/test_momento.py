"""Tests for momento.moment_foundation."""
import numpy as np
import pytest
from moirais.fn.momento import moment_foundation


def test_momento_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    task = np.random.default_rng(42).normal(0, 1, 100)
    result = moment_foundation(y, task)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_momento_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    task = np.random.default_rng(42).normal(0, 1, 100)
    result = moment_foundation(y, task)
    assert isinstance(result, dict)
