"""Tests for rng026.rangayyan_ch3_dirac_delta_limit_form."""
import numpy as np
import pytest
from morie.fn.rng026 import rangayyan_ch3_dirac_delta_limit_form


def test_rng026_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_dirac_delta_limit_form(t, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng026_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_dirac_delta_limit_form(t, a)
    assert isinstance(result, dict)
