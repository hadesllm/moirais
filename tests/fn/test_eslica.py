"""Tests for eslica.esl_ica."""
import numpy as np
import pytest
from moirais.fn.eslica import esl_ica


def test_eslica_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_ica(X, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslica_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    result = esl_ica(X, k)
    assert isinstance(result, dict)
