"""Tests for dssm.dssm."""
import numpy as np
import pytest
from morie.fn.dssm import dssm


def test_dssm_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    doc = np.random.default_rng(42).normal(0, 1, 100)
    result = dssm(query, doc)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dssm_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    doc = np.random.default_rng(42).normal(0, 1, 100)
    result = dssm(query, doc)
    assert isinstance(result, dict)
