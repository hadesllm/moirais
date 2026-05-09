"""Tests for moirais.fn.ssie — Sample size for equivalence."""

import pytest

from moirais.fn.ssie import sample_size_equivalence


class TestSampleSizeEquivalence:
    def test_basic(self):
        res = sample_size_equivalence(0.0, 0.5, 1.0)
        assert res.estimate > 10

    def test_tighter_margin_more_n(self):
        r1 = sample_size_equivalence(0.0, 0.5, 1.0)
        r2 = sample_size_equivalence(0.0, 0.2, 1.0)
        assert r2.estimate > r1.estimate

    def test_delta_exceeds_margin(self):
        with pytest.raises(ValueError):
            sample_size_equivalence(0.6, 0.5, 1.0)
