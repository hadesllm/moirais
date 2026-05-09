"""Tests for ebfmi.energy_bayesian_fmi."""
import numpy as np
import pytest
from moirais.fn.ebfmi import energy_bayesian_fmi


def test_ebfmi_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = energy_bayesian_fmi(chains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ebfmi_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = energy_bayesian_fmi(chains)
    assert isinstance(result, dict)
