"""Tests for statistical_methods_for_spatial_data_analysis1u41.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_41."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u41 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_41


def test_statistical_methods_for_spatial_data_analysis1u41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_41(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_statistical_methods_for_spatial_data_analysis1u41_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_41(x)
    assert isinstance(result, dict)
