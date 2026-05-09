"""Tests for famusm.family_based_assoc."""
import numpy as np
import pytest
from moirais.fn.famusm import family_based_assoc


def test_famusm_basic():
    """Test basic functionality."""
    trios = np.random.default_rng(42).normal(0, 1, 100)
    result = family_based_assoc(trios)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_famusm_edge():
    """Test edge cases."""
    trios = np.random.default_rng(42).normal(0, 1, 100)
    result = family_based_assoc(trios)
    assert isinstance(result, dict)
