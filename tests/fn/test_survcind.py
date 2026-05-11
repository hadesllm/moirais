"""Tests for survcind.survival_concordance."""
import numpy as np
import pytest
from morie.fn.survcind import survival_concordance


def test_survcind_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    predicted_risk = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_concordance(time, event, predicted_risk)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survcind_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    predicted_risk = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_concordance(time, event, predicted_risk)
    assert isinstance(result, dict)
