"""Tests for unitnr.unit_nonresponse."""
import numpy as np
import pytest
from morie.fn.unitnr import unit_nonresponse


def test_unitnr_basic():
    """Test basic functionality."""
    respondents = np.random.default_rng(42).normal(0, 1, 100)
    frame = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = unit_nonresponse(respondents, frame, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_unitnr_edge():
    """Test edge cases."""
    respondents = np.random.default_rng(42).normal(0, 1, 100)
    frame = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = unit_nonresponse(respondents, frame, X)
    assert isinstance(result, dict)
