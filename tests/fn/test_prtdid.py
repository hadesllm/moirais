"""Tests for prtdid.partition_did."""
import numpy as np
import pytest
from moirais.fn.prtdid import partition_did


def test_prtdid_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = partition_did(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prtdid_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = partition_did(y, D, X)
    assert isinstance(result, dict)
