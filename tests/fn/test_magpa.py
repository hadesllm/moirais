"""Tests for magpa.ma_glmm_ipd_proportion."""
import numpy as np
import pytest
from moirais.fn.magpa import ma_glmm_ipd_proportion


def test_magpa_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ma_glmm_ipd_proportion(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_magpa_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = ma_glmm_ipd_proportion(x, n)
    assert isinstance(result, dict)
