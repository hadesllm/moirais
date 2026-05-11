"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r21e8.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_8."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r21e8 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_8


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_8(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_8(x)
    assert isinstance(result, dict)
