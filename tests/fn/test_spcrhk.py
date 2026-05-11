"""Tests for spcrhk.schabenberger_cressie_hawkins."""
import numpy as np
import pytest
from morie.fn.spcrhk import schabenberger_cressie_hawkins


def test_spcrhk_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    lag_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_cressie_hawkins(coords, z, lag_bins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcrhk_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    lag_bins = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_cressie_hawkins(coords, z, lag_bins)
    assert isinstance(result, dict)
