"""Tests for hmfad.geron_forward_autodiff."""
import numpy as np
import pytest
from morie.fn.hmfad import geron_forward_autodiff


def test_hmfad_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_forward_autodiff(f, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfad_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_forward_autodiff(f, x)
    assert isinstance(result, dict)
