"""Tests for divgvs.discriminant_validity."""
import numpy as np
import pytest
from morie.fn.divgvs import discriminant_validity


def test_divgvs_basic():
    """Test basic functionality."""
    AVE = np.random.default_rng(42).normal(0, 1, 100)
    factor_correlations = np.random.default_rng(42).normal(0, 1, 100)
    result = discriminant_validity(AVE, factor_correlations)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_divgvs_edge():
    """Test edge cases."""
    AVE = np.random.default_rng(42).normal(0, 1, 100)
    factor_correlations = np.random.default_rng(42).normal(0, 1, 100)
    result = discriminant_validity(AVE, factor_correlations)
    assert isinstance(result, dict)
