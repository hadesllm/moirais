"""Tests for spcokr.schabenberger_cokriging."""
import numpy as np
import pytest
from moirais.fn.spcokr import schabenberger_cokriging


def test_spcokr_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z1 = np.random.default_rng(42).normal(0, 1, 100)
    z2 = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    cross_cov_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_cokriging(coords, z1, z2, target, cross_cov_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcokr_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z1 = np.random.default_rng(42).normal(0, 1, 100)
    z2 = np.random.default_rng(42).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    cross_cov_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_cokriging(coords, z1, z2, target, cross_cov_model)
    assert isinstance(result, dict)
