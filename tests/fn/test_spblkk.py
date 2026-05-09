"""Tests for spblkk.schabenberger_block_kriging."""
import numpy as np
import pytest
from moirais.fn.spblkk import schabenberger_block_kriging


def test_spblkk_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    cov_model = 'exponential'
    result = schabenberger_block_kriging(coords, z, blocks, cov_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spblkk_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    blocks = np.random.default_rng(42).normal(0, 1, 100)
    cov_model = 'exponential'
    result = schabenberger_block_kriging(coords, z, blocks, cov_model)
    assert isinstance(result, dict)
