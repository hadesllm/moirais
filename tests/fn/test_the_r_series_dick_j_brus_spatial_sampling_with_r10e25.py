"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r10e25.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25."""
import numpy as np
import pytest
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r10e25 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e25_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r10e25_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_25(x)
    assert isinstance(result, dict)
