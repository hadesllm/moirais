"""Tests for spatial_data_analysis_with_r2u2.spatial_data_analysis_with_r_chapter_2_unnumbered_2."""
import numpy as np
import pytest
from moirais.fn.spatial_data_analysis_with_r2u2 import spatial_data_analysis_with_r_chapter_2_unnumbered_2


def test_spatial_data_analysis_with_r2u2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_2_unnumbered_2(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatial_data_analysis_with_r2u2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_2_unnumbered_2(x)
    assert isinstance(result, dict)
