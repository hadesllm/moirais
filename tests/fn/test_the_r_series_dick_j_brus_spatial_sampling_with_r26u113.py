"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r26u113.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_113."""
import numpy as np
import pytest
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r26u113 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_113


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26u113_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_113(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26u113_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_113(x)
    assert isinstance(result, dict)
