"""Tests for moirais.fn.intdt — inter-event interval detection."""
import numpy as np
import pytest

from moirais.fn.intdt import interval_detect, intdt


def test_regular_intervals():
    events = np.arange(0, 5, 0.5)
    result = interval_detect(events, min_interval=0.1)
    assert abs(result.value - 0.5) < 1e-10


def test_filter_small():
    events = np.array([0, 0.05, 0.5, 1.0])
    result = interval_detect(events, min_interval=0.1)
    assert result.extra["n_kept"] == 2


def test_too_few_raises():
    with pytest.raises(ValueError):
        interval_detect([1.0])


def test_alias():
    assert intdt is interval_detect
