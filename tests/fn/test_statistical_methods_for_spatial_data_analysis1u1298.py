"""Tests for statistical_methods_for_spatial_data_analysis1u1298.statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1298."""
import numpy as np
import pytest
from moirais.fn.statistical_methods_for_spatial_data_analysis1u1298 import statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1298


def test_statistical_methods_for_spatial_data_analysis1u1298_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1298(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_statistical_methods_for_spatial_data_analysis1u1298_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = statistical_methods_for_spatial_data_analysis_chapter_1_unnumbered_1298(x)
    assert isinstance(result, dict)
