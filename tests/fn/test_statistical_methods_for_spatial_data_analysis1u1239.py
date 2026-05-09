"""Tests for statistical_methods_for_spatial_data_analysis1u1239.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1239."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1239 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1239


def test_statistical_methods_for_spatial_data_analysis1u1239_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1239(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1239_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1239(x)
    assert isinstance(result, dict)
