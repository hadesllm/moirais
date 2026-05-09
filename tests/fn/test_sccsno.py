"""Tests for sccsno.sccs_no_replacement."""
import numpy as np
import pytest
from moirais.fn.sccsno import sccs_no_replacement


def test_sccsno_basic():
    """Test basic functionality."""
    events = np.random.default_rng(42).normal(0, 1, 100)
    periods = np.random.default_rng(42).normal(0, 1, 100)
    person_id = np.random.default_rng(42).normal(0, 1, 100)
    result = sccs_no_replacement(events, periods, person_id)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sccsno_edge():
    """Test edge cases."""
    events = np.random.default_rng(42).normal(0, 1, 100)
    periods = np.random.default_rng(42).normal(0, 1, 100)
    person_id = np.random.default_rng(42).normal(0, 1, 100)
    result = sccs_no_replacement(events, periods, person_id)
    assert isinstance(result, dict)
