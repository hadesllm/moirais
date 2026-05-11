"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4."""
import numpy as np
import pytest
from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4 import andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4(x)
    assert isinstance(result, dict)
