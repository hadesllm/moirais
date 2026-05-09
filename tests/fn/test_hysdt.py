"""Tests for moirais.fn.hysdt — hysteresis detector."""
import numpy as np
import pytest

from moirais.fn.hysdt import hysteresis_detect, hysdt


def test_single_event():
    x = np.array([0, 0, 0.8, 0.9, 0.5, 0.2, 0, 0])
    result = hysteresis_detect(x, low=0.3, high=0.7)
    assert result.extra["n_events"] == 1
    assert len(result.extra["onsets"]) == 1


def test_no_events():
    x = np.zeros(50)
    result = hysteresis_detect(x, low=0.3, high=0.7)
    assert result.extra["n_events"] == 0


def test_durations():
    x = np.array([0, 0.8, 0.9, 0.5, 0.2, 0, 0.8, 0.6, 0.1])
    result = hysteresis_detect(x, low=0.3, high=0.7)
    assert len(result.extra["durations"]) == result.extra["n_events"]


def test_alias():
    assert hysdt is hysteresis_detect
