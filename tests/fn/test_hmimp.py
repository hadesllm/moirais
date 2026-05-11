"""Tests for hmimp.geron_imputation_median."""
import numpy as np
import pytest
from morie.fn.hmimp import geron_imputation_median


def test_hmimp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_imputation_median(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmimp_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_imputation_median(X)
    assert isinstance(result, dict)
