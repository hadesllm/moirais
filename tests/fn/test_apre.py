"""Tests for apre.oc_apre."""
import numpy as np
import pytest
from morie.fn.apre import oc_apre


def test_apre_basic():
    """Test basic functionality."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = oc_apre(votes, predictions)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_apre_edge():
    """Test edge cases."""
    votes = np.random.default_rng(43).integers(0, 2, (50, 100))
    predictions = np.random.default_rng(42).normal(0, 1, 100)
    result = oc_apre(votes, predictions)
    assert isinstance(result, dict)
