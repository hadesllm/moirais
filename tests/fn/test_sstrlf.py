"""Tests for sstrlf.surv_truncation_left."""
import numpy as np
import pytest
from morie.fn.sstrlf import surv_truncation_left


def test_sstrlf_basic():
    """Test basic functionality."""
    entry = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = surv_truncation_left(entry, time, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sstrlf_edge():
    """Test edge cases."""
    entry = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = surv_truncation_left(entry, time, event)
    assert isinstance(result, dict)
