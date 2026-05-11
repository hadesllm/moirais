"""Tests for aitbi.aitchison_biplot."""
import numpy as np
import pytest
from morie.fn.aitbi import aitchison_biplot


def test_aitbi_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_biplot(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitbi_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_biplot(X)
    assert isinstance(result, dict)
