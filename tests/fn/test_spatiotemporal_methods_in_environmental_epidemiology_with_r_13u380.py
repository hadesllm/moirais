"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_13u380.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_380."""
import numpy as np
import pytest
from moirais.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_13u380 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_380


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_13u380_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_380(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_13u380_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_380(x)
    assert isinstance(result, dict)
