"""Tests for density_estimation_for_statistics_and_data_analysis_silverma2u15.density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_15."""
import numpy as np
import pytest
from moirais.fn.density_estimation_for_statistics_and_data_analysis_silverma2u15 import density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_15


def test_density_estimation_for_statistics_and_data_analysis_silverma2u15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_15(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_density_estimation_for_statistics_and_data_analysis_silverma2u15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_15(x)
    assert isinstance(result, dict)
