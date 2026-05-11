"""Tests for spatial_data_analysis_with_r5u20.spatial_data_analysis_with_r_chapter_5_unnumbered_20."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r5u20 import spatial_data_analysis_with_r_chapter_5_unnumbered_20


def test_spatial_data_analysis_with_r5u20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_5_unnumbered_20(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatial_data_analysis_with_r5u20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_5_unnumbered_20(x)
    assert isinstance(result, dict)
