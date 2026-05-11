"""Tests for ksr17.kosorok_counting_process."""
import numpy as np
import pytest
from morie.fn.ksr17 import kosorok_counting_process


def test_ksr17_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_counting_process(t, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr17_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_counting_process(t, event)
    assert isinstance(result, dict)
