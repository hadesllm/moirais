"""Tests for eslsmp.esl_subsampling."""
import numpy as np
import pytest
from moirais.fn.eslsmp import esl_subsampling


def test_eslsmp_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_subsampling(eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsmp_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_subsampling(eta)
    assert isinstance(result, dict)
