"""Tests for morie.fn.sntcn — sentence concurrency."""

import pytest
import pandas as pd
from morie.fn.sntcn import sentence_concurrency
from morie.fn._containers import DescriptiveResult


class TestSentenceConcurrency:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"sentence_type": ["concurrent", "consecutive", "concurrent", "consecutive"],
                           "sentence_days": [30, 90, 45, 120]})
        result = sentence_concurrency(df)
        assert isinstance(result, DescriptiveResult)

    def test_two_types(self):
        df = pd.DataFrame({"sentence_type": ["concurrent", "consecutive"], "sentence_days": [10, 20]})
        result = sentence_concurrency(df)
        assert len(result.value) == 2
