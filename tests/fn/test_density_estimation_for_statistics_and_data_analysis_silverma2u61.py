"""Tests for density_estimation_for_statistics_and_data_analysis_silverma2u61.density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_61."""
import numpy as np
import pytest
from morie.fn.density_estimation_for_statistics_and_data_analysis_silverma2u61 import density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_61


def test_density_estimation_for_statistics_and_data_analysis_silverma2u61_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_61(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_density_estimation_for_statistics_and_data_analysis_silverma2u61_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_61(x)
    assert isinstance(result, dict)
