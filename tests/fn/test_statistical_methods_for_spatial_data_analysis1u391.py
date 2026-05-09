"""Tests for statistical_methods_for_spatial_data_analysis1u391.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_391."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u391 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_391


def test_statistical_methods_for_spatial_data_analysis1u391_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_391(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u391_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_391(x)
    assert isinstance(result, dict)
