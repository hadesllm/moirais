"""Tests for survci2.uno_concordance."""
import numpy as np
import pytest
from morie.fn.survci2 import uno_concordance


def test_survci2_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    predicted_risk = np.random.default_rng(42).normal(0, 1, 100)
    result = uno_concordance(time, event, predicted_risk)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survci2_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    predicted_risk = np.random.default_rng(42).normal(0, 1, 100)
    result = uno_concordance(time, event, predicted_risk)
    assert isinstance(result, dict)
