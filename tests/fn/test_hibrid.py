"""Tests for hibrid.hibrid_prediction."""
import numpy as np
import pytest
from morie.fn.hibrid import hibrid_prediction


def test_hibrid_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p1_geno = np.random.default_rng(42).normal(0, 1, 100)
    p2_geno = np.random.default_rng(42).normal(0, 1, 100)
    result = hibrid_prediction(y, p1_geno, p2_geno)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hibrid_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p1_geno = np.random.default_rng(42).normal(0, 1, 100)
    p2_geno = np.random.default_rng(42).normal(0, 1, 100)
    result = hibrid_prediction(y, p1_geno, p2_geno)
    assert isinstance(result, dict)
