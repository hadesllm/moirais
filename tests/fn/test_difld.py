"""Tests for difld -- Lord's chi-square DIF."""
from moirais.fn.difld import dif_lord_chisq
from moirais.fn._containers import DIFResult


class TestDifLord:
    def test_no_dif(self):
        ref = {f"item_{j}": {"a": 1.0, "b": 0.0} for j in range(5)}
        foc = {f"item_{j}": {"a": 1.0, "b": 0.0} for j in range(5)}
        result = dif_lord_chisq(ref, foc)
        assert isinstance(result, DIFResult)
        assert result.method == "Lord"
        assert len(result.flagged) == 0

    def test_large_dif(self):
        ref = {f"item_{j}": {"a": 1.0, "b": 0.0} for j in range(5)}
        foc = {f"item_{j}": {"a": 1.0, "b": 3.0} for j in range(5)}
        result = dif_lord_chisq(ref, foc)
        assert len(result.flagged) > 0
