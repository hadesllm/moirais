"""Tests for spcovf.schabenberger_covariance_function."""
import numpy as np
import pytest
from morie.fn.spcovf import schabenberger_covariance_function


def test_spcovf_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_covariance_function(coords, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcovf_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_covariance_function(coords, z)
    assert isinstance(result, dict)
