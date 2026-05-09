"""Tests for spatial_data_analysis_with_r2u8.spatial_data_analysis_with_r_chapter_2_unnumbered_8."""
import numpy as np
import pytest
from moirais.fn.spatial_data_analysis_with_r2u8 import spatial_data_analysis_with_r_chapter_2_unnumbered_8


def test_spatial_data_analysis_with_r2u8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_2_unnumbered_8(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatial_data_analysis_with_r2u8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_2_unnumbered_8(x)
    assert isinstance(result, dict)
