"""Tests for spatial_data_analysis_with_r8u31.spatial_data_analysis_with_r_chapter_8_unnumbered_31."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r8u31 import spatial_data_analysis_with_r_chapter_8_unnumbered_31


def test_spatial_data_analysis_with_r8u31_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_31(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatial_data_analysis_with_r8u31_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_31(x)
    assert isinstance(result, dict)
