"""Tests for servR.serendipity."""
import numpy as np
import pytest
from morie.fn.servR import serendipity


def test_servR_basic():
    """Test basic functionality."""
    pred = np.random.default_rng(42).normal(0, 1, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    result = serendipity(pred, baseline, relevant)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_servR_edge():
    """Test edge cases."""
    pred = np.random.default_rng(42).normal(0, 1, 100)
    baseline = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    result = serendipity(pred, baseline, relevant)
    assert isinstance(result, dict)
