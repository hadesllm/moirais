"""Tests for statistical_methods_for_spatial_data_analysis1u1081.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1081."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1081 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1081


def test_statistical_methods_for_spatial_data_analysis1u1081_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1081(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u1081_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1081(x)
    assert isinstance(result, dict)
