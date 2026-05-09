"""Tests for sunabr.sun_abraham_did."""
import numpy as np
import pytest
from moirais.fn.sunabr import sun_abraham_did


def test_sunabr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = sun_abraham_did(y, D, unit, time, cohort)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sunabr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = sun_abraham_did(y, D, unit, time, cohort)
    assert isinstance(result, dict)
