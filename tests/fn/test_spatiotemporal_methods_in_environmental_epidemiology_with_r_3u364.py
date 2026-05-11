"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_3u364.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_364."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_3u364 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_364


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_3u364_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_364(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_3u364_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_364(x)
    assert isinstance(result, dict)
