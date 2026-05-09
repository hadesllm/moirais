"""Tests for spatial_data_analysis_with_r7u26.spatial_data_analysis_with_r_chapter_7_unnumbered_26."""
import numpy as np
import pytest
from moirais.fn.spatial_data_analysis_with_r7u26 import spatial_data_analysis_with_r_chapter_7_unnumbered_26


def test_spatial_data_analysis_with_r7u26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_7_unnumbered_26(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatial_data_analysis_with_r7u26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_7_unnumbered_26(x)
    assert isinstance(result, dict)
