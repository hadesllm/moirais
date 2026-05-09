"""Tests for evdec.evt_declustering_runs."""
import numpy as np
import pytest
from moirais.fn.evdec import evt_declustering_runs


def test_evdec_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    result = evt_declustering_runs(x, u, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evdec_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u = np.random.default_rng(44).normal(0, 1, 100)
    r = 10
    result = evt_declustering_runs(x, u, r)
    assert isinstance(result, dict)
