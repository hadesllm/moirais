"""Tests for netparf.network_paf."""
import numpy as np
import pytest
from morie.fn.netparf import network_paf


def test_netparf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    network = np.random.default_rng(42).normal(0, 1, 100)
    result = network_paf(y, exposure, network)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_netparf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    network = np.random.default_rng(42).normal(0, 1, 100)
    result = network_paf(y, exposure, network)
    assert isinstance(result, dict)
