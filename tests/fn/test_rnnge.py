"""Tests for rnnge.rnn_genomic."""
import numpy as np
import pytest
from moirais.fn.rnnge import rnn_genomic


def test_rnnge_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    result = rnn_genomic(x, y, markers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rnnge_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    result = rnn_genomic(x, y, markers)
    assert isinstance(result, dict)
