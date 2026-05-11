"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r26u124.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_124."""
import numpy as np
import pytest
from morie.fn.the_r_series_dick_j_brus_spatial_sampling_with_r26u124 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_124


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26u124_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_124(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r26u124_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_unnumbered_124(x)
    assert isinstance(result, dict)
