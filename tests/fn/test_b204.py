"""Tests for b204.burkov_lm_ch2_trigram_count."""
import numpy as np
import pytest
from morie.fn.b204 import burkov_lm_ch2_trigram_count


def test_b204_basic():
    """Test basic functionality."""
    t_i = np.random.default_rng(42).normal(0, 1, 100)
    t_im1 = np.random.default_rng(42).normal(0, 1, 100)
    t_im2 = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch2_trigram_count(t_i, t_im1, t_im2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b204_edge():
    """Test edge cases."""
    t_i = np.random.default_rng(42).normal(0, 1, 100)
    t_im1 = np.random.default_rng(42).normal(0, 1, 100)
    t_im2 = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch2_trigram_count(t_i, t_im1, t_im2)
    assert isinstance(result, dict)
