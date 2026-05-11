"""Tests for cipsc.caliper_psm."""
import numpy as np
import pytest
from morie.fn.cipsc import caliper_psm


def test_cipsc_basic():
    """Test basic functionality."""
    e_score = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = caliper_psm(e_score, T, caliper)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cipsc_edge():
    """Test edge cases."""
    e_score = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    caliper = np.random.default_rng(42).normal(0, 1, 100)
    result = caliper_psm(e_score, T, caliper)
    assert isinstance(result, dict)
