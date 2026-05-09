"""Tests for moirais.fn.sntpl — sentence by plea."""

import pytest
import pandas as pd
from moirais.fn.sntpl import sentence_plea
from moirais.fn._containers import DescriptiveResult


class TestSentencePlea:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"plea_type": ["guilty", "trial", "guilty", "trial"],
                           "sentence_days": [30, 90, 45, 120]})
        result = sentence_plea(df)
        assert isinstance(result, DescriptiveResult)

    def test_trial_penalty_visible(self):
        df = pd.DataFrame({"plea_type": ["guilty"] * 10 + ["trial"] * 10,
                           "sentence_days": [30] * 10 + [90] * 10})
        result = sentence_plea(df)
        means = dict(zip(result.value["plea"], result.value["mean"]))
        assert means["trial"] > means["guilty"]
