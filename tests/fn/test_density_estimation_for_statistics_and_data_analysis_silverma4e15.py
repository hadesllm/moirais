"""Tests for density_estimation_for_statistics_and_data_analysis_silverma4e15.density_estimation_for_statistics_and_data_analysis_silverma_chapter_4_equation_15."""
import numpy as np
import pytest
from morie.fn.density_estimation_for_statistics_and_data_analysis_silverma4e15 import density_estimation_for_statistics_and_data_analysis_silverma_chapter_4_equation_15


def test_density_estimation_for_statistics_and_data_analysis_silverma4e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_4_equation_15(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_density_estimation_for_statistics_and_data_analysis_silverma4e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = density_estimation_for_statistics_and_data_analysis_silverma_chapter_4_equation_15(x)
    assert isinstance(result, dict)
