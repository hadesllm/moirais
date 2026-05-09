"""Tests for statistical_methods_for_spatial_data_analysis1u1749.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1749."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1749 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1749


def test_statistical_methods_for_spatial_data_analysis1u1749_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1749(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u1749_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1749(x)
    assert isinstance(result, dict)
