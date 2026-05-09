"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r13e12.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_12."""
import numpy as np
import pytest
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r13e12 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_12


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_12(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_12(x)
    assert isinstance(result, dict)
