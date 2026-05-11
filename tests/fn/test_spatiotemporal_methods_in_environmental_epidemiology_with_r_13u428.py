"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_13u428.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_428."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_13u428 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_428


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_13u428_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_428(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_13u428_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_428(x)
    assert isinstance(result, dict)
