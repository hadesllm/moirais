"""Tests for vpc.variance_partition_coefficient."""
import numpy as np
import pytest
from moirais.fn.vpc import variance_partition_coefficient


def test_vpc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_partition_coefficient(y, cluster, sigma2_u)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vpc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    sigma2_u = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_partition_coefficient(y, cluster, sigma2_u)
    assert isinstance(result, dict)
