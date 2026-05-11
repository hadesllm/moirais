"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r11u81.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_81."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r11u81 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_81


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11u81_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_81(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11u81_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_81(x)
    assert isinstance(result, dict)
