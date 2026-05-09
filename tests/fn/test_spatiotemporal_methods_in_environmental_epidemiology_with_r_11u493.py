"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_11u493.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_493."""
import numpy as np
import pytest
from moirais.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_11u493 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_493


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_11u493_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_493(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_11u493_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_11_unnumbered_493(x)
    assert isinstance(result, dict)
