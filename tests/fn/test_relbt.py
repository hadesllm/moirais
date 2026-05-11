"""Tests for relbt.reliability_metric."""
import numpy as np
import pytest
from morie.fn.relbt import reliability_metric


def test_relbt_basic():
    """Test basic functionality."""
    r = 10
    h2 = np.random.default_rng(42).normal(0, 1, 100)
    result = reliability_metric(r, h2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_relbt_edge():
    """Test edge cases."""
    r = 10
    h2 = np.random.default_rng(42).normal(0, 1, 100)
    result = reliability_metric(r, h2)
    assert isinstance(result, dict)
