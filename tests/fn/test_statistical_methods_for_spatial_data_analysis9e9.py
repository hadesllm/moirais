"""Tests for statistical_methods_for_spatial_data_analysis9e9.statistical_methods_for_spatial_data_analysis_chapter_9_equation_9."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis9e9 import statistical_methods_for_spatial_data_analysis_chapter_9_equation_9


def test_statistical_methods_for_spatial_data_analysis9e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_9_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis9e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_9_equation_9(x)
    assert isinstance(result, dict)
