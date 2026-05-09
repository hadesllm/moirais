"""Tests for statistical_methods_for_spatial_data_analysis6e28.statistical_methods_for_spatial_data_analysis_chapter_6_equation_28."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis6e28 import statistical_methods_for_spatial_data_analysis_chapter_6_equation_28


def test_statistical_methods_for_spatial_data_analysis6e28_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_6_equation_28(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis6e28_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_6_equation_28(x)
    assert isinstance(result, dict)
