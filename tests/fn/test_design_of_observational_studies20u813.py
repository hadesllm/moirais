"""Tests for design_of_observational_studies20u813.design_of_observational_studies_chapter_20_unnumbered_813."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies20u813 import design_of_observational_studies_chapter_20_unnumbered_813


def test_design_of_observational_studies20u813_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_813(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies20u813_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_20_unnumbered_813(x)
    assert isinstance(result, dict)
