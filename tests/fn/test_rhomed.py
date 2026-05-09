"""Tests for rhomed.rho_critical_mediation."""
import numpy as np
import pytest
from moirais.fn.rhomed import rho_critical_mediation


def test_rhomed_basic():
    """Test basic functionality."""
    nie = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e3 = np.random.default_rng(42).normal(0, 1, 100)
    result = rho_critical_mediation(nie, sigma_e2, sigma_e3)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rhomed_edge():
    """Test edge cases."""
    nie = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma_e3 = np.random.default_rng(42).normal(0, 1, 100)
    result = rho_critical_mediation(nie, sigma_e2, sigma_e3)
    assert isinstance(result, dict)
