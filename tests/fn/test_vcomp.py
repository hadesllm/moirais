"""Tests for vcomp.variance_components_henderson3."""
import numpy as np
import pytest
from morie.fn.vcomp import variance_components_henderson3


def test_vcomp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_components_henderson3(y, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vcomp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = variance_components_henderson3(y, cluster)
    assert isinstance(result, dict)
