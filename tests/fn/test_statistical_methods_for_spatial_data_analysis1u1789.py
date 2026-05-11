"""Tests for statistical_methods_for_spatial_data_analysis1u1789.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1789."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u1789 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1789


def test_statistical_methods_for_spatial_data_analysis1u1789_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1789(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1789_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1789(x)
    assert isinstance(result, dict)
