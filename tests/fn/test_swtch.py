"""Tests for morie.fn.swtch -- label / one-hot encoding."""

import pandas as pd
from morie.fn.swtch import encode_labels, swtch
from morie.fn._containers import DescriptiveResult


class TestSwtch:
    def test_alias(self):
        assert swtch is encode_labels

    def test_label_encoding(self):
        df = pd.DataFrame({"category": ["a", "b", "c", "a", "b"]})
        result = encode_labels(df, column="category", method="label")
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_categories"] == 3

    def test_onehot(self):
        df = pd.DataFrame({"category": ["x", "y", "x"], "val": [1, 2, 3]})
        result = encode_labels(df, column="category", method="onehot")
        encoded = result.value
        assert "x" in encoded.columns
        assert "y" in encoded.columns
