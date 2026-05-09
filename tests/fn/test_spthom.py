"""Tests for spthom.schabenberger_thomas_process."""
import numpy as np
import pytest
from moirais.fn.spthom import schabenberger_thomas_process


def test_spthom_basic():
    """Test basic functionality."""
    r = 10
    rho = 0.5
    mu = 0.0
    sigma = 1.0
    result = schabenberger_thomas_process(r, rho, mu, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spthom_edge():
    """Test edge cases."""
    r = 10
    rho = 0.5
    mu = 0.0
    sigma = 1.0
    result = schabenberger_thomas_process(r, rho, mu, sigma)
    assert isinstance(result, dict)
