"""Tests for mhst1.mantel_haenszel_or."""
import numpy as np
import pytest
from moirais.fn.mhst1 import mantel_haenszel_or


def test_mhst1_basic():
    """Test basic functionality."""
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = mantel_haenszel_or(strata)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mhst1_edge():
    """Test edge cases."""
    strata = np.random.default_rng(42).normal(0, 1, 100)
    result = mantel_haenszel_or(strata)
    assert isinstance(result, dict)
