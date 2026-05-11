"""Tests for rgerp.rangayyan_erp_features."""
import numpy as np
import pytest
from morie.fn.rgerp import rangayyan_erp_features


def test_rgerp_basic():
    """Test basic functionality."""
    erp = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_erp_features(erp, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgerp_edge():
    """Test edge cases."""
    erp = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_erp_features(erp, fs)
    assert isinstance(result, dict)
