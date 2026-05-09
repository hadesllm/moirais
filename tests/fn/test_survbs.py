"""Tests for survbs.survival_bootstrap_se."""
import numpy as np
import pytest
from moirais.fn.survbs import survival_bootstrap_se


def test_survbs_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = survival_bootstrap_se(time, event, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survbs_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = survival_bootstrap_se(time, event, B)
    assert isinstance(result, dict)
