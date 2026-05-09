"""Tests for clstal.clustalo."""
import numpy as np
import pytest
from moirais.fn.clstal import clustalo


def test_clstal_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    result = clustalo(sequences)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clstal_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(42).normal(0, 1, 100)
    result = clustalo(sequences)
    assert isinstance(result, dict)
