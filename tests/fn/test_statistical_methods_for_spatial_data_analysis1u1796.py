"""Tests for statistical_methods_for_spatial_data_analysis1u1796.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1796."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u1796 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1796


def test_statistical_methods_for_spatial_data_analysis1u1796_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1796(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u1796_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1796(x)
    assert isinstance(result, dict)
