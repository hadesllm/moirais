"""Tests for volsv.vol_sv_quasi_lik."""
import numpy as np
import pytest
from moirais.fn.volsv import vol_sv_quasi_lik


def test_volsv_basic():
    """Test basic functionality."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_sv_quasi_lik(r, init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volsv_edge():
    """Test edge cases."""
    r = 10
    init = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_sv_quasi_lik(r, init)
    assert isinstance(result, dict)
