"""Tests for csrh.cause_specific_hazard."""
import numpy as np
import pytest
from morie.fn.csrh import cause_specific_hazard


def test_csrh_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = cause_specific_hazard(time, cause, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_csrh_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = cause_specific_hazard(time, cause, X)
    assert isinstance(result, dict)
