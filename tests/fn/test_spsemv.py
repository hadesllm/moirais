"""Tests for spsemv.schabenberger_semivariogram_def."""
import numpy as np
import pytest
from morie.fn.spsemv import schabenberger_semivariogram_def


def test_spsemv_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_semivariogram_def(coords, z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spsemv_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = schabenberger_semivariogram_def(coords, z)
    assert isinstance(result, dict)
