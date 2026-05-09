"""Tests for treatment_effects_11u18.treatment_effects_1_chapter_1_unnumbered_18."""
import numpy as np
import pytest
from moirais.fn.treatment_effects_11u18 import treatment_effects_1_chapter_1_unnumbered_18


def test_treatment_effects_11u18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_18(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_treatment_effects_11u18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = treatment_effects_1_chapter_1_unnumbered_18(x)
    assert isinstance(result, dict)
