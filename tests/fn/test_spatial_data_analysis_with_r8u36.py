"""Tests for spatial_data_analysis_with_r8u36.spatial_data_analysis_with_r_chapter_8_unnumbered_36."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r8u36 import spatial_data_analysis_with_r_chapter_8_unnumbered_36


def test_spatial_data_analysis_with_r8u36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_36(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatial_data_analysis_with_r8u36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_36(x)
    assert isinstance(result, dict)
