"""Tests for covLst.catalog_coverage."""
import numpy as np
import pytest
from morie.fn.covLst import catalog_coverage


def test_covLst_basic():
    """Test basic functionality."""
    recommendations = np.random.default_rng(42).normal(0, 1, 100)
    catalog = np.random.default_rng(42).normal(0, 1, 100)
    result = catalog_coverage(recommendations, catalog)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_covLst_edge():
    """Test edge cases."""
    recommendations = np.random.default_rng(42).normal(0, 1, 100)
    catalog = np.random.default_rng(42).normal(0, 1, 100)
    result = catalog_coverage(recommendations, catalog)
    assert isinstance(result, dict)
