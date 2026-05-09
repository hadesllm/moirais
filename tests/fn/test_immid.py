"""Tests for immid.index_moderated_mediation."""
import numpy as np
import pytest
from moirais.fn.immid import index_moderated_mediation


def test_immid_basic():
    """Test basic functionality."""
    a1 = np.random.default_rng(42).normal(0, 1, 100)
    a3 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = index_moderated_mediation(a1, a3, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_immid_edge():
    """Test edge cases."""
    a1 = np.random.default_rng(42).normal(0, 1, 100)
    a3 = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = index_moderated_mediation(a1, a3, b)
    assert isinstance(result, dict)
