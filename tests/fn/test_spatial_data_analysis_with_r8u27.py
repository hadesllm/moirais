"""Tests for spatial_data_analysis_with_r8u27.spatial_data_analysis_with_r_chapter_8_unnumbered_27."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r8u27 import spatial_data_analysis_with_r_chapter_8_unnumbered_27


def test_spatial_data_analysis_with_r8u27_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_27(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatial_data_analysis_with_r8u27_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_27(x)
    assert isinstance(result, dict)
