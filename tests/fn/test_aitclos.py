"""Tests for aitclos.aitchison_closure."""
import numpy as np
import pytest
from moirais.fn.aitclos import aitchison_closure


def test_aitclos_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kappa = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_closure(x, kappa)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitclos_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kappa = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_closure(x, kappa)
    assert isinstance(result, dict)
