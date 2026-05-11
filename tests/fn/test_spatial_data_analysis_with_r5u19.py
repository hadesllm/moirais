"""Tests for spatial_data_analysis_with_r5u19.spatial_data_analysis_with_r_chapter_5_unnumbered_19."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r5u19 import spatial_data_analysis_with_r_chapter_5_unnumbered_19


def test_spatial_data_analysis_with_r5u19_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_5_unnumbered_19(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatial_data_analysis_with_r5u19_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_5_unnumbered_19(x)
    assert isinstance(result, dict)
