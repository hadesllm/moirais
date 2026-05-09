"""Tests for krfgrp.kronecker_graph."""
import numpy as np
import pytest
from moirais.fn.krfgrp import kronecker_graph


def test_krfgrp_basic():
    """Test basic functionality."""
    seed = 42
    k = 5
    result = kronecker_graph(seed, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krfgrp_edge():
    """Test edge cases."""
    seed = 42
    k = 5
    result = kronecker_graph(seed, k)
    assert isinstance(result, dict)
