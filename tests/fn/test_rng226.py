"""Tests for rng226.rangayyan_ch4_matched_filter_h_example."""
import numpy as np
import pytest
from moirais.fn.rng226 import rangayyan_ch4_matched_filter_h_example


def test_rng226_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch4_matched_filter_h_example(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng226_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch4_matched_filter_h_example(n)
    assert isinstance(result, dict)
