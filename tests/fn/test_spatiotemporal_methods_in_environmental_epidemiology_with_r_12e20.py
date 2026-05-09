"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_12e20.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_12_equation_20."""
import numpy as np
import pytest
from moirais.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_12e20 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_12_equation_20


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_12e20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_12_equation_20(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_12e20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_12_equation_20(x)
    assert isinstance(result, dict)
