"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r12u11.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_unnumbered_11."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r12u11 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_unnumbered_11


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12u11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_unnumbered_11(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12u11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_unnumbered_11(x)
    assert isinstance(result, dict)
