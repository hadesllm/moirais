"""Tests for moirais.fn.beast -- Jukes-Cantor mutation rate."""

from moirais.fn.beast import jukes_cantor_rate, beast
from moirais.fn._containers import DescriptiveResult


class TestBeast:
    def test_alias(self):
        assert beast is jukes_cantor_rate

    def test_identical_sequences(self):
        r = jukes_cantor_rate(["ACGT" * 10, "ACGT" * 10])
        assert isinstance(r, DescriptiveResult)
        assert r.value["mean_distance"] == 0.0

    def test_different_sequences(self):
        r = jukes_cantor_rate(["AAAA", "CCCC", "GGGG"])
        D = r.value["distance_matrix"]
        assert D[0, 1] > 0
        assert r.value["n_pairs"] == 3
