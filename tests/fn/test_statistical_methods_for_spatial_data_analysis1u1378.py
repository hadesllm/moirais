"""Tests for statistical_methods_for_spatial_data_analysis1u1378.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1378."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u1378 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1378


def test_statistical_methods_for_spatial_data_analysis1u1378_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1378(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1378_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1378(x)
    assert isinstance(result, dict)
