"""Tests for mhrate.mantel_haenszel_rate."""
import numpy as np
import pytest
from morie.fn.mhrate import mantel_haenszel_rate


def test_mhrate_basic():
    """Test basic functionality."""
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = mantel_haenszel_rate(strata)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mhrate_edge():
    """Test edge cases."""
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = mantel_haenszel_rate(strata)
    assert isinstance(result, dict)
