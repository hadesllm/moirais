"""Tests for survtdc.time_dep_concordance."""
import numpy as np
import pytest
from morie.fn.survtdc import time_dep_concordance


def test_survtdc_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    marker = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = time_dep_concordance(time, event, marker, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survtdc_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    marker = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = time_dep_concordance(time, event, marker, t)
    assert isinstance(result, dict)
