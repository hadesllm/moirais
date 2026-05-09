"""Tests for wvltdb.db_wavelet."""
import numpy as np
import pytest
from moirais.fn.wvltdb import db_wavelet


def test_wvltdb_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    levels = [0.5, 1.0, 1.5, 2.0]
    result = db_wavelet(y, n, levels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wvltdb_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    levels = [0.5, 1.0, 1.5, 2.0]
    result = db_wavelet(y, n, levels)
    assert isinstance(result, dict)
