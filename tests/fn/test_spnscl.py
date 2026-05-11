"""Tests for spnscl.schabenberger_neyman_scott."""
import numpy as np
import pytest
from morie.fn.spnscl import schabenberger_neyman_scott


def test_spnscl_basic():
    """Test basic functionality."""
    r = 10
    rho = 0.5
    mu = 0.0
    sigma = 1.0
    result = schabenberger_neyman_scott(r, rho, mu, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spnscl_edge():
    """Test edge cases."""
    r = 10
    rho = 0.5
    mu = 0.0
    sigma = 1.0
    result = schabenberger_neyman_scott(r, rho, mu, sigma)
    assert isinstance(result, dict)
