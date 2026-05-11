"""Tests for diopT.farey_seq."""
import numpy as np
import pytest
from morie.fn.diopT import farey_seq


def test_diopT_basic():
    """Test basic functionality."""
    n = 100
    result = farey_seq(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diopT_edge():
    """Test edge cases."""
    n = 100
    result = farey_seq(n)
    assert isinstance(result, dict)
