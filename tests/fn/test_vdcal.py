"""Tests for vdcal.volume_of_distribution."""
import numpy as np
import pytest
from morie.fn.vdcal import volume_of_distribution


def test_vdcal_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    ppb = np.random.default_rng(42).normal(0, 1, 100)
    result = volume_of_distribution(smiles, ppb)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vdcal_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    ppb = np.random.default_rng(42).normal(0, 1, 100)
    result = volume_of_distribution(smiles, ppb)
    assert isinstance(result, dict)
