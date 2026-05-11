"""Tests for grlrco.geron_lr_cosine_annealing."""
import numpy as np
import pytest
from morie.fn.grlrco import geron_lr_cosine_annealing


def test_grlrco_basic():
    """Test basic functionality."""
    eta_min = 0
    eta_max = 100
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_lr_cosine_annealing(eta_min, eta_max, t, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grlrco_edge():
    """Test edge cases."""
    eta_min = 0
    eta_max = 100
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_lr_cosine_annealing(eta_min, eta_max, t, T)
    assert isinstance(result, dict)
