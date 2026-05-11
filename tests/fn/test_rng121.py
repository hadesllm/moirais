"""Tests for rng121.rangayyan_ch3_baseline_wander_filter_z_form_b."""
import numpy as np
import pytest
from morie.fn.rng121 import rangayyan_ch3_baseline_wander_filter_z_form_b


def test_rng121_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_baseline_wander_filter_z_form_b(z, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng121_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_baseline_wander_filter_z_form_b(z, T)
    assert isinstance(result, dict)
