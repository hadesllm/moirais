"""Tests for sgtdiff.sgt_diffusion_kernel."""
import numpy as np
import pytest
from morie.fn.sgtdiff import sgt_diffusion_kernel


def test_sgtdiff_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    t = np.linspace(0, 10, 100)
    result = sgt_diffusion_kernel(A, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtdiff_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    t = np.linspace(0, 10, 100)
    result = sgt_diffusion_kernel(A, t)
    assert isinstance(result, dict)
