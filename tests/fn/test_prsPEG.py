"""Tests for prsPEG.peg_parser."""
import numpy as np
import pytest
from morie.fn.prsPEG import peg_parser


def test_prsPEG_basic():
    """Test basic functionality."""
    grammar = np.random.default_rng(42).normal(0, 1, 100)
    input = np.random.default_rng(42).normal(0, 1, 100)
    result = peg_parser(grammar, input)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsPEG_edge():
    """Test edge cases."""
    grammar = np.random.default_rng(42).normal(0, 1, 100)
    input = np.random.default_rng(42).normal(0, 1, 100)
    result = peg_parser(grammar, input)
    assert isinstance(result, dict)
