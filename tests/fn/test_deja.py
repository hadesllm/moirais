"""Tests for moirais.fn.deja -- near-duplicate detection."""

import numpy as np
import pandas as pd
from moirais.fn.deja import detect_duplicates, deja
from moirais.fn._containers import DescriptiveResult


class TestDeja:
    def test_alias(self):
        assert deja is detect_duplicates

    def test_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = detect_duplicates(df, threshold=0.99)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_no_duplicates(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"a": rng.normal(0, 10, 20), "b": rng.normal(0, 10, 20)})
        result = detect_duplicates(df, threshold=0.99999)
        assert result.value == 0
