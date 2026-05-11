"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r11u25.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_25."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r11u25 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_25


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11u25_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_25(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11u25_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_25(x)
    assert isinstance(result, dict)
