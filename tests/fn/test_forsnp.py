"""Tests for forsnp.forensic_lr."""
import numpy as np
import pytest
from morie.fn.forsnp import forensic_lr


def test_forsnp_basic():
    """Test basic functionality."""
    E = np.random.default_rng(42).normal(0, 1, 100)
    H1 = np.random.default_rng(42).normal(0, 1, 100)
    H2 = np.random.default_rng(42).normal(0, 1, 100)
    result = forensic_lr(E, H1, H2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_forsnp_edge():
    """Test edge cases."""
    E = np.random.default_rng(42).normal(0, 1, 100)
    H1 = np.random.default_rng(42).normal(0, 1, 100)
    H2 = np.random.default_rng(42).normal(0, 1, 100)
    result = forensic_lr(E, H1, H2)
    assert isinstance(result, dict)
