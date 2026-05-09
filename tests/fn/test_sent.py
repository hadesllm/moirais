"""Tests for moirais.fn.sent -- Sentence length stats."""

import numpy as np
import pytest
from moirais.fn.sent import sentence_stats, sent
from moirais.fn._containers import DescriptiveResult


class TestSent:
    def test_alias(self):
        assert sent is sentence_stats

    def test_basic_stats(self):
        result = sentence_stats(np.array([12, 24, 36, 48, 60]))
        assert isinstance(result, DescriptiveResult)
        assert result.value == pytest.approx(36.0)
        assert result.extra["median"] == pytest.approx(36.0)

    def test_mean(self):
        result = sentence_stats(np.array([12, 24, 36, 48, 60]))
        assert result.extra["mean"] == pytest.approx(36.0)
