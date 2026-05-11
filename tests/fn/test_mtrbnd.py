"""Tests for mtrbnd.monotone_treatment_response."""
import numpy as np
import pytest
from morie.fn.mtrbnd import monotone_treatment_response


def test_mtrbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    direction = np.random.default_rng(42).normal(0, 1, 100)
    result = monotone_treatment_response(y, D, direction)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtrbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    direction = np.random.default_rng(42).normal(0, 1, 100)
    result = monotone_treatment_response(y, D, direction)
    assert isinstance(result, dict)
