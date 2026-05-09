"""Tests for grdpmf.geron_ddpm_forward_process."""
import numpy as np
import pytest
from moirais.fn.grdpmf import geron_ddpm_forward_process


def test_grdpmf_basic():
    """Test basic functionality."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddpm_forward_process(x0, t, alpha_bar)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdpmf_edge():
    """Test edge cases."""
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    alpha_bar = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ddpm_forward_process(x0, t, alpha_bar)
    assert isinstance(result, dict)
