"""Tests for morie.fn.dmxlr -- DNA mixture likelihood ratio."""

import numpy as np
import pytest
from morie.fn.dmxlr import dmxlr


class TestDmxlr:
    def test_single_locus_match(self):
        mix = [{"A", "B"}]
        sus = [("A", "B")]
        freq = [{"A": 0.3, "B": 0.2}]
        res = dmxlr(mix, sus, freq, n_contributors=1)
        assert res.extra["LR"] >= 1.0

    def test_suspect_excluded(self):
        mix = [{"A", "B"}]
        sus = [("C", "D")]
        freq = [{"A": 0.3, "B": 0.2, "C": 0.1, "D": 0.1}]
        res = dmxlr(mix, sus, freq, n_contributors=2)
        assert res.extra["per_locus_lr"][0] == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            dmxlr([{"A"}], [("A", "B"), ("C", "D")], [{"A": 0.5}])
