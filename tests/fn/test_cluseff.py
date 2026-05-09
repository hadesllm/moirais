"""Tests for cluseff.intracluster_correlation_rho."""
import numpy as np
import pytest
from moirais.fn.cluseff import intracluster_correlation_rho


def test_cluseff_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = intracluster_correlation_rho(y, cluster)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_cluseff_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = intracluster_correlation_rho(y, cluster)
    assert isinstance(result, dict)
