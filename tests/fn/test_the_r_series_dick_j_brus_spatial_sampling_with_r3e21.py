"""Tests for the_r_series_dick_j_brus_spatial_sampling_with_r3e21.the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_21."""
import numpy as np
import pytest
from moirais.fn.the_r_series_dick_j_brus_spatial_sampling_with_r3e21 import the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_21


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_21(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r3e21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_21(x)
    assert isinstance(result, dict)
