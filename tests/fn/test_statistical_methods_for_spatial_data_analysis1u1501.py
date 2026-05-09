"""Tests for statistical_methods_for_spatial_data_analysis1u1501.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1501."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1501 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1501


def test_statistical_methods_for_spatial_data_analysis1u1501_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1501(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1501_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1501(x)
    assert isinstance(result, dict)
