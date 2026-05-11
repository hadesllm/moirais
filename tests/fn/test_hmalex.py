"""Tests for hmalex.geron_alexnet."""
import numpy as np
import pytest
from morie.fn.hmalex import geron_alexnet


def test_hmalex_basic():
    """Test basic functionality."""
    n_classes = 3
    result = geron_alexnet(n_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmalex_edge():
    """Test edge cases."""
    n_classes = 3
    result = geron_alexnet(n_classes)
    assert isinstance(result, dict)
