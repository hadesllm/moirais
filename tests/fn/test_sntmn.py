"""Tests for morie.fn.sntmn — mandatory minimum sentence."""

import pytest
import pandas as pd
from morie.fn.sntmn import sentence_mandatory_min
from morie.fn._containers import DescriptiveResult


class TestSentenceMandatoryMin:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"offense": ["A", "B", "A", "B"],
                           "sentence_days": [30, 90, 60, 120],
                           "mandatory_min_days": [30, 60, 30, 60]})
        result = sentence_mandatory_min(df)
        assert isinstance(result, DescriptiveResult)

    def test_all_at_min(self):
        df = pd.DataFrame({"offense": ["A", "A"], "sentence_days": [30, 30], "mandatory_min_days": [30, 30]})
        result = sentence_mandatory_min(df)
        assert result.extra["pct_at_minimum"] == pytest.approx(1.0)
