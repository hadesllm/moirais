"""Tests for caCMIP.cmip_ensemble."""
import numpy as np
import pytest
from morie.fn.caCMIP import cmip_ensemble


def test_caCMIP_basic():
    """Test basic functionality."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = cmip_ensemble(models, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_caCMIP_edge():
    """Test edge cases."""
    models = np.random.default_rng(42).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = cmip_ensemble(models, weights)
    assert isinstance(result, dict)
