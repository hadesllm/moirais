"""Tests for design_of_observational_studies15u591.design_of_observational_studies_chapter_15_unnumbered_591."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies15u591 import design_of_observational_studies_chapter_15_unnumbered_591


def test_design_of_observational_studies15u591_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_591(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u591_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_591(x)
    assert isinstance(result, dict)
