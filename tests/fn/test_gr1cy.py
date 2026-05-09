"""Tests for gr1cy.geron_1cycle_schedule."""
import numpy as np
import pytest
from moirais.fn.gr1cy import geron_1cycle_schedule


def test_gr1cy_basic():
    """Test basic functionality."""
    eta_min = 0
    eta_max = 100
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_1cycle_schedule(eta_min, eta_max, t, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gr1cy_edge():
    """Test edge cases."""
    eta_min = 0
    eta_max = 100
    t = np.linspace(0, 10, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_1cycle_schedule(eta_min, eta_max, t, T)
    assert isinstance(result, dict)
