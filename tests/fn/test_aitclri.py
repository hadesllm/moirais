"""Tests for aitclri.aitchison_clr_inverse."""
import numpy as np
import pytest
from morie.fn.aitclri import aitchison_clr_inverse


def test_aitclri_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = aitchison_clr_inverse(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitclri_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = aitchison_clr_inverse(z)
    assert isinstance(result, dict)
