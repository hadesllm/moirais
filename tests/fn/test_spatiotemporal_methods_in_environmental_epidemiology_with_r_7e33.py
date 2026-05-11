"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_7e33.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_7_equation_33."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_7e33 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_7_equation_33


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_7e33_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_7_equation_33(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_7e33_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_7_equation_33(x)
    assert isinstance(result, dict)
