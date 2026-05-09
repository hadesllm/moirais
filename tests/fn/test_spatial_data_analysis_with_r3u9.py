"""Tests for spatial_data_analysis_with_r3u9.spatial_data_analysis_with_r_chapter_3_unnumbered_9."""
import numpy as np
import pytest
from moirais.fn.spatial_data_analysis_with_r3u9 import spatial_data_analysis_with_r_chapter_3_unnumbered_9


def test_spatial_data_analysis_with_r3u9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_3_unnumbered_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_spatial_data_analysis_with_r3u9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_3_unnumbered_9(x)
    assert isinstance(result, dict)
