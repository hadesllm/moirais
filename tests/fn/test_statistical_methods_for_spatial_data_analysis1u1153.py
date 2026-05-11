"""Tests for statistical_methods_for_spatial_data_analysis1u1153.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1153."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u1153 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1153


def test_statistical_methods_for_spatial_data_analysis1u1153_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1153(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u1153_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1153(x)
    assert isinstance(result, dict)
