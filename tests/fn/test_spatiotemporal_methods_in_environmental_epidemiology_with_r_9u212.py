"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_9u212.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_212."""
import numpy as np
import pytest
from morie.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_9u212 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_212


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_9u212_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_212(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_9u212_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_212(x)
    assert isinstance(result, dict)
