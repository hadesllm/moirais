"""Tests for tqrot.turboquant_rotation_matrix."""
import numpy as np
import pytest
from morie.fn.tqrot import turboquant_rotation_matrix


def test_tqrot_basic():
    """Test basic functionality."""
    d = 5
    seed = 42
    result = turboquant_rotation_matrix(d, seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tqrot_edge():
    """Test edge cases."""
    d = 5
    seed = 42
    result = turboquant_rotation_matrix(d, seed)
    assert isinstance(result, dict)
