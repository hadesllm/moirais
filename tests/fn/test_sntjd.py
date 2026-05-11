"""Tests for morie.fn.sntjd — sentence judicial variation."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.sntjd import sentence_judicial
from morie.fn._containers import DescriptiveResult


class TestSentenceJudicial:

    def test_returns_descriptive(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"judge_id": np.repeat(["J1", "J2", "J3"], 20),
                           "sentence_days": np.concatenate([rng.normal(30, 5, 20),
                                                            rng.normal(60, 5, 20),
                                                            rng.normal(90, 5, 20)])})
        result = sentence_judicial(df)
        assert isinstance(result, DescriptiveResult)
        assert "icc" in result.extra

    def test_icc_bounded(self):
        df = pd.DataFrame({"judge_id": ["A", "A", "B", "B"], "sentence_days": [10, 10, 100, 100]})
        result = sentence_judicial(df)
        assert 0 <= result.extra["icc"] <= 1
