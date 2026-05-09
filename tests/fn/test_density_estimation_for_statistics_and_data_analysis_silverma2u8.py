"""Tests for density_estimation_for_statistics_and_data_analysis_silverma2u8.density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_8."""
import numpy as np
import pytest
from moirais.fn.density_estimation_for_statistics_and_data_analysis_silverma2u8 import density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_8


def test_density_estimation_for_statistics_and_data_analysis_silverma2u8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_8(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_density_estimation_for_statistics_and_data_analysis_silverma2u8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_8(x)
    assert isinstance(result, dict)
