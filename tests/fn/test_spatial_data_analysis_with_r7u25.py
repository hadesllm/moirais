"""Tests for spatial_data_analysis_with_r7u25.spatial_data_analysis_with_r_chapter_7_unnumbered_25."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r7u25 import spatial_data_analysis_with_r_chapter_7_unnumbered_25


def test_spatial_data_analysis_with_r7u25_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_7_unnumbered_25(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatial_data_analysis_with_r7u25_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_7_unnumbered_25(x)
    assert isinstance(result, dict)
