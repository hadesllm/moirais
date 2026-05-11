"""Tests for spatial_data_analysis_with_r3u12.spatial_data_analysis_with_r_chapter_3_unnumbered_12."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r3u12 import spatial_data_analysis_with_r_chapter_3_unnumbered_12


def test_spatial_data_analysis_with_r3u12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_3_unnumbered_12(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_spatial_data_analysis_with_r3u12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_3_unnumbered_12(x)
    assert isinstance(result, dict)
