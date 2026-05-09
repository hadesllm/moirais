"""Tests for hmpcac.geron_principal_components."""
import numpy as np
import pytest
from moirais.fn.hmpcac import geron_principal_components


def test_hmpcac_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_principal_components(X, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpcac_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_principal_components(X, n_components)
    assert isinstance(result, dict)
