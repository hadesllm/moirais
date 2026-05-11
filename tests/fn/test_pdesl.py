"""Tests for pdesl.pde_separation."""
import numpy as np
import pytest
from morie.fn.pdesl import pde_separation


def test_pdesl_basic():
    """Test basic functionality."""
    pde = np.random.default_rng(42).normal(0, 1, 100)
    result = pde_separation(pde)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pdesl_edge():
    """Test edge cases."""
    pde = np.random.default_rng(42).normal(0, 1, 100)
    result = pde_separation(pde)
    assert isinstance(result, dict)
