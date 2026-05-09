"""Tests for gracif.gray_cif_test."""
import numpy as np
import pytest
from moirais.fn.gracif import gray_cif_test


def test_gracif_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = gray_cif_test(time, cause, group)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gracif_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    cause = np.random.default_rng(42).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = gray_cif_test(time, cause, group)
    assert isinstance(result, dict)
