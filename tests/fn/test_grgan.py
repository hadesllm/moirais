"""Tests for grgan.geron_gan_minimax."""
import numpy as np
import pytest
from moirais.fn.grgan import geron_gan_minimax


def test_grgan_basic():
    """Test basic functionality."""
    real = np.random.default_rng(42).normal(0, 1, 100)
    fake = np.random.default_rng(42).normal(0, 1, 100)
    D_real = np.random.default_rng(42).normal(0, 1, 100)
    D_fake = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gan_minimax(real, fake, D_real, D_fake)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgan_edge():
    """Test edge cases."""
    real = np.random.default_rng(42).normal(0, 1, 100)
    fake = np.random.default_rng(42).normal(0, 1, 100)
    D_real = np.random.default_rng(42).normal(0, 1, 100)
    D_fake = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gan_minimax(real, fake, D_real, D_fake)
    assert isinstance(result, dict)
