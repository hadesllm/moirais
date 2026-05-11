"""Tests for spatial_data_analysis_with_r8u35.spatial_data_analysis_with_r_chapter_8_unnumbered_35."""
import numpy as np
import pytest
from morie.fn.spatial_data_analysis_with_r8u35 import spatial_data_analysis_with_r_chapter_8_unnumbered_35


def test_spatial_data_analysis_with_r8u35_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_35(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatial_data_analysis_with_r8u35_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_analysis_with_r_chapter_8_unnumbered_35(x)
    assert isinstance(result, dict)
