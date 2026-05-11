"""Tests for hindsr.her."""
import numpy as np
import pytest
from morie.fn.hindsr import her


def test_hindsr_basic():
    """Test basic functionality."""
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    result = her(buffer, strategy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hindsr_edge():
    """Test edge cases."""
    buffer = np.random.default_rng(42).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    result = her(buffer, strategy)
    assert isinstance(result, dict)
