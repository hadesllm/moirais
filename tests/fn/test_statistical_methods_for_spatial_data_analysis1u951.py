"""Tests for statistical_methods_for_spatial_data_analysis1u951.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_951."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u951 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_951


def test_statistical_methods_for_spatial_data_analysis1u951_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_951(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u951_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_951(x)
    assert isinstance(result, dict)
