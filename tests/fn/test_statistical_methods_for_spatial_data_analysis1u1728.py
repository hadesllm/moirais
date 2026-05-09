"""Tests for statistical_methods_for_spatial_data_analysis1u1728.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1728."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1728 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1728


def test_statistical_methods_for_spatial_data_analysis1u1728_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1728(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u1728_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1728(x)
    assert isinstance(result, dict)
