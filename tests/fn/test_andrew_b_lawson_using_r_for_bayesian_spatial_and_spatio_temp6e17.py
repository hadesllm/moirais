"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e17.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_17."""
import numpy as np
import pytest
from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e17 import andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_17


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_17(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_17(x)
    assert isinstance(result, dict)
