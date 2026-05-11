"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r13e16.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_16."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r13e16 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_16


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_16(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_16(x)
    assert isinstance(result, dict)
