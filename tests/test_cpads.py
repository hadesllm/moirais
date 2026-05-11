"""Tests for morie.cpads — CPADS data contract and canonicalization."""

import numpy as np
import pandas as pd
import pytest

from morie.cpads import (
    CPADS_RAW_COLUMN_MAP,
    CPADS_REQUIRED_VARIABLES,
    canonicalize_cpads_frame,
    cpads_contract,
    has_raw_cpads_columns,
    infer_file_format,
    missing_cpads_variables,
    validate_cpads_frame,
)


class TestContract:
    def test_contract_returns_dict(self):
        c = cpads_contract()
        assert isinstance(c, dict)
        assert "required_variables" in c
        assert "raw_column_map" in c
        assert "source_kind" in c

    def test_contract_variables_match_constant(self):
        c = cpads_contract()
        assert c["required_variables"] == list(CPADS_REQUIRED_VARIABLES)

    def test_contract_column_map_match(self):
        c = cpads_contract()
        assert c["raw_column_map"] == dict(CPADS_RAW_COLUMN_MAP)


class TestMissingVariables:
    def test_all_present(self):
        assert missing_cpads_variables(CPADS_REQUIRED_VARIABLES) == []

    def test_some_missing(self):
        missing = missing_cpads_variables(["weight", "gender"])
        assert "alcohol_past12m" in missing
        assert "weight" not in missing
        assert "gender" not in missing

    def test_none_present(self):
        missing = missing_cpads_variables([])
        assert len(missing) == len(CPADS_REQUIRED_VARIABLES)


class TestValidateFrame:
    def _canonical_frame(self, n: int = 10) -> pd.DataFrame:
        """Build a minimal valid canonical CPADS frame."""
        rng = np.random.default_rng(42)
        data = {var: rng.integers(0, 5, size=n) for var in CPADS_REQUIRED_VARIABLES}
        data["weight"] = rng.random(n)
        return pd.DataFrame(data)

    def test_valid_frame_returns_empty(self):
        df = self._canonical_frame()
        result = validate_cpads_frame(df)
        assert result == []

    def test_missing_col_strict_raises(self):
        df = self._canonical_frame().drop(columns=["gender"])
        with pytest.raises(ValueError, match="missing required"):
            validate_cpads_frame(df, strict=True)

    def test_missing_col_nonstrict(self):
        df = self._canonical_frame().drop(columns=["gender"])
        missing = validate_cpads_frame(df, strict=False)
        assert "gender" in missing


class TestHasRawColumns:
    def _raw_frame(self, weight_col: str = "wtpumf") -> pd.DataFrame:
        raw_cols = set(CPADS_RAW_COLUMN_MAP.values())
        data = {col: [1, 2, 3] for col in raw_cols}
        if weight_col != "wtpumf":
            data.pop("wtpumf", None)
            data[weight_col] = [1, 2, 3]
        return pd.DataFrame(data)

    def test_raw_pumf(self):
        assert has_raw_cpads_columns(self._raw_frame("wtpumf"))

    def test_raw_wtdf(self):
        assert has_raw_cpads_columns(self._raw_frame("wtdf"))

    def test_canonical_is_not_raw(self):
        df = pd.DataFrame({var: [1] for var in CPADS_REQUIRED_VARIABLES})
        assert not has_raw_cpads_columns(df)


class TestCanonicalizeFrame:
    def _raw_frame(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "wtpumf": rng.random(n),
            "alc05": rng.choice([1, 2, 98, 99], size=n),
            "alc12_30d_prev_total": rng.choice([0, 1], size=n),
            "alc12_30d_prev": rng.choice([0, 1], size=n),
            "can05": rng.choice([1, 2, 98, 99], size=n),
            "age_groups": rng.choice([1, 2, 3, 98], size=n),
            "dvdemq01": rng.choice([1, 2, 98], size=n),
            "region": rng.choice([1, 2, 3, 4, 5], size=n),
            "hwbq02": rng.choice([1, 2, 3, 4, 98], size=n),
            "hwbq01": rng.choice([1, 2, 3, 4, 98], size=n),
            "ebac_tot": rng.random(n) * 10,
            "ebac_legal": rng.random(n) * 5,
        })

    def test_produces_canonical_columns(self):
        raw = self._raw_frame()
        out = canonicalize_cpads_frame(raw)
        for var in CPADS_REQUIRED_VARIABLES:
            assert var in out.columns

    def test_does_not_modify_input(self):
        raw = self._raw_frame()
        original_cols = list(raw.columns)
        _ = canonicalize_cpads_frame(raw)
        assert list(raw.columns) == original_cols

    def test_alcohol_recoding(self):
        raw = self._raw_frame()
        out = canonicalize_cpads_frame(raw)
        # Values should be 0, 1, or NaN
        valid = out["alcohol_past12m"].dropna().unique()
        assert set(valid).issubset({0, 1})

    def test_cannabis_recoding(self):
        raw = self._raw_frame()
        out = canonicalize_cpads_frame(raw)
        valid = out["cannabis_any_use"].dropna().unique()
        assert set(valid).issubset({0, 1})

    def test_weight_is_numeric(self):
        raw = self._raw_frame()
        out = canonicalize_cpads_frame(raw)
        assert pd.api.types.is_numeric_dtype(out["weight"])

    def test_canonical_frame_passes_through(self):
        """If already canonical, should return copy."""
        rng = np.random.default_rng(42)
        data = {var: rng.integers(0, 5, size=10) for var in CPADS_REQUIRED_VARIABLES}
        data["weight"] = rng.random(10)
        df = pd.DataFrame(data)
        out = canonicalize_cpads_frame(df)
        assert list(out.columns) == list(df.columns)

    def test_nonraw_noncanonocal_raises(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with pytest.raises(ValueError):
            canonicalize_cpads_frame(df)


class TestInferFileFormat:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("data.csv", "csv"),
            ("path/to/file.xlsx", "excel"),
            ("path/to/file.xls", "excel"),
            ("path/to/data.rds", "rds"),
        ],
    )
    def test_known_formats(self, path, expected):
        assert infer_file_format(path) == expected

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            infer_file_format("data.parquet")
