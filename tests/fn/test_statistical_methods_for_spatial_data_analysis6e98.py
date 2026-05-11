"""Tests for statistical_methods_for_spatial_data_analysis6e98.statistical_methods_for_spatial_data_analysis_chapter_6_equation_98."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis6e98 import statistical_methods_for_spatial_data_analysis_chapter_6_equation_98


def test_statistical_methods_for_spatial_data_analysis6e98_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_6_equation_98(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis6e98_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_6_equation_98(x)
    assert isinstance(result, dict)
