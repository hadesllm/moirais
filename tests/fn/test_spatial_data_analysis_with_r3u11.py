"""Tests for spatial_data_analysis_with_r3u11.spatial_data_analysis_with_r_chapter_3_unnumbered_11."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r3u11 import spatial_data_analysis_with_r_chapter_3_unnumbered_11


def test_spatial_data_analysis_with_r3u11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_3_unnumbered_11(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatial_data_analysis_with_r3u11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_3_unnumbered_11(x)
    assert isinstance(result, dict)
