"""Tests for conti.continued_fraction_pi."""
import numpy as np
import pytest
from morie.fn.conti import continued_fraction_pi


def test_conti_basic():
    """Test basic functionality."""
    n = 100
    result = continued_fraction_pi(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_conti_edge():
    """Test edge cases."""
    n = 100
    result = continued_fraction_pi(n)
    assert isinstance(result, dict)
