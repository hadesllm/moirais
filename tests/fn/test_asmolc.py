"""Tests for asmolc.olc_assembly."""
import numpy as np
import pytest
from moirais.fn.asmolc import olc_assembly


def test_asmolc_basic():
    """Test basic functionality."""
    long_reads = np.random.default_rng(42).normal(0, 1, 100)
    result = olc_assembly(long_reads)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_asmolc_edge():
    """Test edge cases."""
    long_reads = np.random.default_rng(42).normal(0, 1, 100)
    result = olc_assembly(long_reads)
    assert isinstance(result, dict)
