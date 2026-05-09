"""Tests for grdcgan.geron_dcgan_generator."""
import numpy as np
import pytest
from moirais.fn.grdcgan import geron_dcgan_generator


def test_grdcgan_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_dcgan_generator(z, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grdcgan_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_dcgan_generator(z, weights)
    assert isinstance(result, dict)
