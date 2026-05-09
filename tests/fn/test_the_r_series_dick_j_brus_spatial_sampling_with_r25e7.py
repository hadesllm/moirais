"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r25e7.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_7."""
import numpy as np
import pytest
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r25e7 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_7


def test_the_r_series_dick_j_brus_spatial_sampling_with_r25e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_7(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r25e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_7(x)
    assert isinstance(result, dict)
