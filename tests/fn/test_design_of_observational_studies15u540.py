"""Tests for design_of_observational_studies15u540.design_of_observational_studies_chapter_15_unnumbered_540."""
import numpy as np
import pytest
from moirais.fn.design_of_observational_studies15u540 import design_of_observational_studies_chapter_15_unnumbered_540


def test_design_of_observational_studies15u540_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_540(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_design_of_observational_studies15u540_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = design_of_observational_studies_chapter_15_unnumbered_540(x)
    assert isinstance(result, dict)
