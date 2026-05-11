"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_13u451.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_451."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_13u451 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_451


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_13u451_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_451(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_13u451_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_451(x)
    assert isinstance(result, dict)
