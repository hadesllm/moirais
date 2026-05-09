"""Tests for moirais.fn.oqntl — Quantiles for a numeric column."""

from moirais.fn.oqntl import otis_quantiles


class TestOtisQuantiles:
    def test_returns_dict(self, otis_df):
        result = otis_quantiles(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_quantiles(otis_df)
        for k in ("quantiles", "n", "column"):
            assert k in result

    def test_default_probs(self, otis_df):
        result = otis_quantiles(otis_df)
        assert set(result["quantiles"].keys()) == {0.25, 0.5, 0.75}

    def test_monotonic(self, otis_df):
        result = otis_quantiles(otis_df)
        q = result["quantiles"]
        assert q[0.25] <= q[0.5] <= q[0.75]

    def test_custom_probs(self, otis_df):
        result = otis_quantiles(otis_df, probs=[0.1, 0.9])
        assert set(result["quantiles"].keys()) == {0.1, 0.9}
