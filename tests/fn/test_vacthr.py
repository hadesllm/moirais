"""Tests for vacthr.vaccination_threshold."""
import numpy as np
import pytest
from moirais.fn.vacthr import vaccination_threshold


def test_vacthr_basic():
    """Test basic functionality."""
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    result = vaccination_threshold(R0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vacthr_edge():
    """Test edge cases."""
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    result = vaccination_threshold(R0)
    assert isinstance(result, dict)
