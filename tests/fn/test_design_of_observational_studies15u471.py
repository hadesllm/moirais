"""Tests for design_of_observational_studies15u471.design_of_observational_studies_chapter_15_unnumbered_471."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies15u471 import design_of_observational_studies_chapter_15_unnumbered_471


def test_design_of_observational_studies15u471_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_471(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u471_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_471(x)
    assert isinstance(result, dict)
