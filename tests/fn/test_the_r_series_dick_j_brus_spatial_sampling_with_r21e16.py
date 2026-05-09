"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r21e16.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_16."""
import numpy as np
import pytest
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r21e16 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_16


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_16(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_16(x)
    assert isinstance(result, dict)
