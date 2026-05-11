"""Tests for emdtsm.emd_decomposition."""
import numpy as np
import pytest
from morie.fn.emdtsm import emd_decomposition


def test_emdtsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_imf = np.random.default_rng(42).normal(0, 1, 100)
    result = emd_decomposition(y, max_imf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_emdtsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_imf = np.random.default_rng(42).normal(0, 1, 100)
    result = emd_decomposition(y, max_imf)
    assert isinstance(result, dict)
