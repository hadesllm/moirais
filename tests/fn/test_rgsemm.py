"""Tests for rgsemm.rangayyan_spec_error_meas."""
import numpy as np
import pytest
from morie.fn.rgsemm import rangayyan_spec_error_meas


def test_rgsemm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    p = 5
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spec_error_meas(x, fs, p, seg_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgsemm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    p = 5
    seg_len = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_spec_error_meas(x, fs, p, seg_len)
    assert isinstance(result, dict)
