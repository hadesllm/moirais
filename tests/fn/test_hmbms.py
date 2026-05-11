"""Tests for hmbms.geron_beam_search."""
import numpy as np
import pytest
from morie.fn.hmbms import geron_beam_search


def test_hmbms_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    src = np.random.default_rng(42).normal(0, 1, 100)
    beam_width = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_beam_search(model, src, beam_width)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbms_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    src = np.random.default_rng(42).normal(0, 1, 100)
    beam_width = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_beam_search(model, src, beam_width)
    assert isinstance(result, dict)
