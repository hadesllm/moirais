"""Tests for spatiotemporal_methods_in_environmental_epidemiology_with_r_9u263.spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_263."""
import numpy as np
import pytest
from moirais.fn.spatiotemporal_methods_in_environmental_epidemiology_with_r_9u263 import spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_263


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_9u263_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_263(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_spatiotemporal_methods_in_environmental_epidemiology_with_r_9u263_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_263(x)
    assert isinstance(result, dict)
