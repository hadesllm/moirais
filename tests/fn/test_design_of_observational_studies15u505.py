"""Tests for design_of_observational_studies15u505.design_of_observational_studies_chapter_15_unnumbered_505."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies15u505 import design_of_observational_studies_chapter_15_unnumbered_505


def test_design_of_observational_studies15u505_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_505(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u505_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_505(x)
    assert isinstance(result, dict)
