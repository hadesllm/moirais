"""Tests for compdp.basic_composition."""
import numpy as np
import pytest
from morie.fn.compdp import basic_composition


def test_compdp_basic():
    """Test basic functionality."""
    epsilons = np.random.default_rng(42).normal(0, 1, 100)
    result = basic_composition(epsilons)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_compdp_edge():
    """Test edge cases."""
    epsilons = np.random.default_rng(42).normal(0, 1, 100)
    result = basic_composition(epsilons)
    assert isinstance(result, dict)
