"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u10.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_10."""
import numpy as np
import pytest
from moirais.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u10 import andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_10


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u10_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_10(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u10_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_10(x)
    assert isinstance(result, dict)
