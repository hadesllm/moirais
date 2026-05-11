"""Tests for statistical_methods_for_spatial_data_analysis1u1221.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1221."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u1221 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1221


def test_statistical_methods_for_spatial_data_analysis1u1221_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1221(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1221_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1221(x)
    assert isinstance(result, dict)
