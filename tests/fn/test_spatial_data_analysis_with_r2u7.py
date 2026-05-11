"""Tests for spatial_data_analysis_with_r2u7.spatial_data_analysis_with_r_chapter_2_unnumbered_7."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r2u7 import spatial_data_analysis_with_r_chapter_2_unnumbered_7


def test_spatial_data_analysis_with_r2u7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_2_unnumbered_7(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatial_data_analysis_with_r2u7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_2_unnumbered_7(x)
    assert isinstance(result, dict)
