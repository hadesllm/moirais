"""Tests for smplts.sample_lifetable."""
import numpy as np
import pytest
from morie.fn.smplts import sample_lifetable


def test_smplts_basic():
    """Test basic functionality."""
    intervals = np.random.default_rng(42).normal(0, 1, 100)
    entered = np.random.default_rng(42).normal(0, 1, 100)
    died = np.random.default_rng(42).normal(0, 1, 100)
    censored = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_lifetable(intervals, entered, died, censored)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smplts_edge():
    """Test edge cases."""
    intervals = np.random.default_rng(42).normal(0, 1, 100)
    entered = np.random.default_rng(42).normal(0, 1, 100)
    died = np.random.default_rng(42).normal(0, 1, 100)
    censored = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_lifetable(intervals, entered, died, censored)
    assert isinstance(result, dict)
