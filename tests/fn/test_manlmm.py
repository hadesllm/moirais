"""Tests for manlmm.ma_network_lme."""
import numpy as np
import pytest
from moirais.fn.manlmm import ma_network_lme


def test_manlmm_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_network_lme(yi, vi, design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_manlmm_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_network_lme(yi, vi, design)
    assert isinstance(result, dict)
