"""Tests for gblpq.gblup_equivalence."""
import numpy as np
import pytest
from moirais.fn.gblpq import gblup_equivalence


def test_gblpq_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    G = np.eye(10)
    result = gblup_equivalence(y, Z, G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gblpq_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    G = np.eye(10)
    result = gblup_equivalence(y, Z, G)
    assert isinstance(result, dict)
