"""Tests for moirais.fn.durdt — duration-based event filtering."""
import numpy as np
import pytest

from moirais.fn.durdt import duration_detect, durdt


def test_filter_short():
    events = [(0, 0.005), (0.1, 0.3), (0.5, 0.6)]
    result = duration_detect(events, min_dur=0.01, max_dur=1.0)
    assert result.value == 2.0


def test_filter_long():
    events = [(0, 0.5), (1, 3)]
    result = duration_detect(events, min_dur=0.01, max_dur=1.0)
    assert result.value == 1.0


def test_all_kept():
    events = [(0, 0.1), (0.5, 0.6)]
    result = duration_detect(events, min_dur=0.01, max_dur=1.0)
    assert result.extra["rejected"] == 0


def test_alias():
    assert durdt is duration_detect
