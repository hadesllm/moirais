"""Tests for hmlcos.geron_cosine_annealing."""
import numpy as np
import pytest
from moirais.fn.hmlcos import geron_cosine_annealing


def test_hmlcos_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    eta_max = 100
    eta_min = 0
    result = geron_cosine_annealing(t, T, eta_max, eta_min)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlcos_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    eta_max = 100
    eta_min = 0
    result = geron_cosine_annealing(t, T, eta_max, eta_min)
    assert isinstance(result, dict)
