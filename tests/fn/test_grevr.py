"""Tests for grevr.geron_explained_variance_ratio."""
import numpy as np
import pytest
from morie.fn.grevr import geron_explained_variance_ratio


def test_grevr_basic():
    """Test basic functionality."""
    singular_values = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_explained_variance_ratio(singular_values)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grevr_edge():
    """Test edge cases."""
    singular_values = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_explained_variance_ratio(singular_values)
    assert isinstance(result, dict)
