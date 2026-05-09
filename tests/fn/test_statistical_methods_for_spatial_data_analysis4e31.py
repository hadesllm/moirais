"""Tests for statistical_methods_for_spatial_data_analysis4e31.statistical_methods_for_spatial_data_analysis_chapter_4_equation_31."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis4e31 import statistical_methods_for_spatial_data_analysis_chapter_4_equation_31


def test_statistical_methods_for_spatial_data_analysis4e31_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_4_equation_31(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis4e31_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_4_equation_31(x)
    assert isinstance(result, dict)
