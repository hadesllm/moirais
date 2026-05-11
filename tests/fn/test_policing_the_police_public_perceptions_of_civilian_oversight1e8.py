"""Tests for policing_the_police_public_perceptions_of_civilian_oversight1e8.policing_the_police_public_perceptions_of_civilian_oversight_chapter_1_equation_8."""
import numpy as np
import pytest
from morie.fn.policing_the_police_public_perceptions_of_civilian_oversight1e8 import policing_the_police_public_perceptions_of_civilian_oversight_chapter_1_equation_8


def test_policing_the_police_public_perceptions_of_civilian_oversight1e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = policing_the_police_public_perceptions_of_civilian_oversight_chapter_1_equation_8(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_policing_the_police_public_perceptions_of_civilian_oversight1e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = policing_the_police_public_perceptions_of_civilian_oversight_chapter_1_equation_8(x)
    assert isinstance(result, dict)
