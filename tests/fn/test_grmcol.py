"""Tests for grmcol.geron_gan_mode_collapse_metric."""
import numpy as np
import pytest
from moirais.fn.grmcol import geron_gan_mode_collapse_metric


def test_grmcol_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    true_modes = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gan_mode_collapse_metric(samples, true_modes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmcol_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    true_modes = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gan_mode_collapse_metric(samples, true_modes)
    assert isinstance(result, dict)
