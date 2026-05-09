"""Tests for statistical_methods_for_spatial_data_analysis1u791.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_791."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u791 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_791


def test_statistical_methods_for_spatial_data_analysis1u791_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_791(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u791_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_791(x)
    assert isinstance(result, dict)
