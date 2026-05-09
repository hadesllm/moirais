"""Tests for btrnd.boot_rng_seeded."""
import numpy as np
import pytest
from moirais.fn.btrnd import boot_rng_seeded


def test_btrnd_basic():
    """Test basic functionality."""
    seed = 42
    result = boot_rng_seeded(seed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btrnd_edge():
    """Test edge cases."""
    seed = 42
    result = boot_rng_seeded(seed)
    assert isinstance(result, dict)
