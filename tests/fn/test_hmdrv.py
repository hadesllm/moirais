"""Tests for hmdrv.geron_diffusion_reverse."""
import numpy as np
import pytest
from moirais.fn.hmdrv import geron_diffusion_reverse


def test_hmdrv_basic():
    """Test basic functionality."""
    x_T = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_diffusion_reverse(x_T, model, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmdrv_edge():
    """Test edge cases."""
    x_T = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_diffusion_reverse(x_T, model, T)
    assert isinstance(result, dict)
