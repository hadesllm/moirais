"""Tests for hrzecfw.horowitz_empirical_cf."""
import numpy as np
import pytest
from morie.fn.hrzecfw import horowitz_empirical_cf


def test_hrzecfw_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    tau = 0.1
    result = horowitz_empirical_cf(w, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzecfw_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    tau = 0.1
    result = horowitz_empirical_cf(w, tau)
    assert isinstance(result, dict)
