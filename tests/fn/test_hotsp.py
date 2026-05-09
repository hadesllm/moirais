"""Tests for moirais.fn.hotsp -- Hot-spot analysis."""

import numpy as np
from moirais.fn.hotsp import hot_spots, hotsp
from moirais.fn._containers import DescriptiveResult


class TestHotsp:
    def test_alias(self):
        assert hotsp is hot_spots

    def test_basic(self):
        result = hot_spots(np.array([1, 1, 1, 2, 2, 3]), threshold=3)
        assert isinstance(result, DescriptiveResult)
        assert 1 in result.extra["hot_spots"]
        assert 2 not in result.extra["hot_spots"]

    def test_counts(self):
        result = hot_spots(np.array([1, 1, 1, 2, 2, 3]), threshold=3)
        assert result.extra["counts"][1] == 3
        assert result.extra["counts"][2] == 2
