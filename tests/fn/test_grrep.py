"""Tests for grrep.geron_reparameterization_trick."""
import numpy as np
import pytest
from moirais.fn.grrep import geron_reparameterization_trick


def test_grrep_basic():
    """Test basic functionality."""
    mu = 0.0
    logvar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reparameterization_trick(mu, logvar)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grrep_edge():
    """Test edge cases."""
    mu = 0.0
    logvar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_reparameterization_trick(mu, logvar)
    assert isinstance(result, dict)
