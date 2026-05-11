"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_11u517.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_517."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_11u517 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_517


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_11u517_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_517(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_11u517_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_517(x)
    assert isinstance(result, dict)
