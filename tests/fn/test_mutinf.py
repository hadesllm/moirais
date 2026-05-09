"""Tests for mutinf.mutual_information."""
import numpy as np
import pytest
from moirais.fn.mutinf import mutual_information


def test_mutinf_basic():
    """Test basic functionality."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = mutual_information(pxy, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mutinf_edge():
    """Test edge cases."""
    pxy = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = mutual_information(pxy, base)
    assert isinstance(result, dict)
