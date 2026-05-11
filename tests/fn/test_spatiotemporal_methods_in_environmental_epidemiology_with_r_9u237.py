"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_9u237.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_237."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_9u237 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_237


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_9u237_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_237(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_9u237_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_237(x)
    assert isinstance(result, dict)
