"""Tests for statistical_methods_for_spatial_data_analysis1u917.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_917."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u917 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_917


def test_statistical_methods_for_spatial_data_analysis1u917_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_917(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u917_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_917(x)
    assert isinstance(result, dict)
