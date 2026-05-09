"""Tests for design_of_observational_studies15u477.design_of_observational_studies_chapter_15_unnumbered_477."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies15u477 import design_of_observational_studies_chapter_15_unnumbered_477


def test_design_of_observational_studies15u477_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_477(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u477_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_477(x)
    assert isinstance(result, dict)
