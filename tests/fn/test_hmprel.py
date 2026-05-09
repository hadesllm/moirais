"""Tests for hmprel.geron_prelu."""
import numpy as np
import pytest
from moirais.fn.hmprel import geron_prelu


def test_hmprel_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = geron_prelu(z, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmprel_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    alpha = 0.05
    result = geron_prelu(z, alpha)
    assert isinstance(result, dict)
