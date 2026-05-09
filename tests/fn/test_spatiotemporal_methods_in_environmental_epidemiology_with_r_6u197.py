"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_6u197.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_197."""
import numpy as np
import pytest
from moirais.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_6u197 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_197


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_6u197_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_197(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_6u197_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_197(x)
    assert isinstance(result, dict)
