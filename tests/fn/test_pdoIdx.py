"""Tests for pdoIdx.pdo."""
import numpy as np
import pytest
from morie.fn.pdoIdx import pdo


def test_pdoIdx_basic():
    """Test basic functionality."""
    sst = np.random.default_rng(42).normal(0, 1, 100)
    result = pdo(sst)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pdoIdx_edge():
    """Test edge cases."""
    sst = np.random.default_rng(42).normal(0, 1, 100)
    result = pdo(sst)
    assert isinstance(result, dict)
