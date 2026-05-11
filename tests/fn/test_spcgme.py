"""Tests for spcgme.spatial_concordance_kappa."""
import numpy as np
import pytest
from morie.fn.spcgme import spatial_concordance_kappa


def test_spcgme_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_concordance_kappa(x, y, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcgme_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_concordance_kappa(x, y, W)
    assert isinstance(result, dict)
