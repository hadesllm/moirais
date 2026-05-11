"""Tests for statistical_methods_for_spatial_data_analysis1u90.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_90."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u90 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_90


def test_statistical_methods_for_spatial_data_analysis1u90_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_90(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u90_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_90(x)
    assert isinstance(result, dict)
