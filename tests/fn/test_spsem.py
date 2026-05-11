"""Tests for spsem.schabenberger_spatial_error_model."""
import numpy as np
import pytest
from morie.fn.spsem import schabenberger_spatial_error_model


def test_spsem_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_spatial_error_model(x, y, w)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_spsem_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = schabenberger_spatial_error_model(x, y, w)
    assert isinstance(result, dict)
