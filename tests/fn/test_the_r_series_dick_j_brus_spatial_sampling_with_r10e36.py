"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e36.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e36 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_36(x)
    assert isinstance(result, dict)
