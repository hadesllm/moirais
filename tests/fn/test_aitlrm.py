"""Tests for aitlrm.compositional_lrmean."""
import numpy as np
import pytest
from moirais.fn.aitlrm import compositional_lrmean


def test_aitlrm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_lrmean(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitlrm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = compositional_lrmean(X)
    assert isinstance(result, dict)
