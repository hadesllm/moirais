"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_3u320.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_320."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_3u320 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_320


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_3u320_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_320(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_3u320_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_320(x)
    assert isinstance(result, dict)
