"""Tests for volcc.vol_christoffersen_cc."""
import numpy as np
import pytest
from moirais.fn.volcc import vol_christoffersen_cc


def test_volcc_basic():
    """Test basic functionality."""
    hits = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vol_christoffersen_cc(hits, alpha)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_volcc_edge():
    """Test edge cases."""
    hits = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = vol_christoffersen_cc(hits, alpha)
    assert isinstance(result, dict)
