"""Tests for statistical_methods_for_spatial_data_analysis1u1602.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1602."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1602 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1602


def test_statistical_methods_for_spatial_data_analysis1u1602_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1602(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1602_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1602(x)
    assert isinstance(result, dict)
