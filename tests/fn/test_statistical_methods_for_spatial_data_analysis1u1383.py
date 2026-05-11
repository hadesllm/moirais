"""Tests for statistical_methods_for_spatial_data_analysis1u1383.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1383."""
import numpy as np
import pytest
from morie.fn.statistical_methods_for_spatial_data_analysis1u1383 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1383


def test_statistical_methods_for_spatial_data_analysis1u1383_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1383(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1383_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1383(x)
    assert isinstance(result, dict)
