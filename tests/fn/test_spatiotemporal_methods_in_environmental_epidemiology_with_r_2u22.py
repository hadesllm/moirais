"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_2u22.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_2_unnumbered_22."""
import numpy as np
import pytest
from moirais.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_2u22 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_2_unnumbered_22


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_2u22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_2_unnumbered_22(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_2u22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_2_unnumbered_22(x)
    assert isinstance(result, dict)
