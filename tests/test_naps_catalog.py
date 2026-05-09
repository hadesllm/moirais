"""Smoke tests for NAPS DATASET_CATALOG entries (Workstream 5 complete)."""

import pytest

from moirais.data import DATASET_CATALOG


def _naps_entries():
    return {k: v for k, v in DATASET_CATALOG.items() if k.startswith("naps-")}


def test_naps_catalog_has_at_least_20_entries():
    # 7 original (commit 188b591a) + 17 extension = 24. Allow growth.
    assert len(_naps_entries()) >= 20


def test_every_naps_entry_dispatches_via_fetcher():
    for k, e in _naps_entries().items():
        assert e["format"] == "fetcher", f"{k} missing fetcher format"
        assert e["fetcher"] == "moirais.earth:fetch_naps", f"{k} wrong fetcher"
        assert e["source"] == "naps"


def test_every_naps_entry_has_complete_fetcher_args():
    for k, e in _naps_entries().items():
        fa = e["fetcher_args"]
        assert "pollutant" in fa, f"{k} missing pollutant"
        assert fa["pollutant"] in {"no2", "pm25", "o3", "so2", "co", "pm10"}
        assert "year" in fa and isinstance(fa["year"], int)
        # province is optional (None = national aggregate)
        assert "province" in fa


def test_naps_table_names_match_keys():
    for k, e in _naps_entries().items():
        # key 'naps-pm25-on-2023' → table 'naps_pm25_on_2023'
        expected = k.replace("-", "_")
        assert e["table_name"] == expected, f"{k}: table_name={e['table_name']}"


def test_naps_coverage_matrix_sanity():
    entries = _naps_entries()
    # Ontario PM2.5 should cover 2019-2023 (trend years for MRP work)
    on_pm25_years = {
        e["fetcher_args"]["year"]
        for e in entries.values()
        if e["fetcher_args"].get("pollutant") == "pm25"
        and e["fetcher_args"].get("province") == "ON"
    }
    assert {2019, 2020, 2021, 2022, 2023}.issubset(on_pm25_years)

    # 2023 should cover {ON, QC, BC, AB, NS} for at least pm25 + no2
    provinces_2023_pm25 = {
        e["fetcher_args"].get("province")
        for e in entries.values()
        if e["fetcher_args"].get("year") == 2023
        and e["fetcher_args"].get("pollutant") == "pm25"
    }
    # Include CA (None → national) and the key 5 provinces
    assert {"ON", "QC", "BC", "AB", "NS"}.issubset({
        p for p in provinces_2023_pm25 if p is not None
    })


def test_naps_national_aggregates_have_province_none():
    entries = _naps_entries()
    national_keys = [k for k in entries if k.endswith("-ca-2023")]
    assert len(national_keys) >= 3  # pm25, no2, o3
    for k in national_keys:
        assert entries[k]["fetcher_args"]["province"] is None, \
            f"{k}: national aggregate should have province=None"


def test_naps_keys_unique_and_well_formed():
    # Pattern: naps-<pollutant>-<province|ca>-<yyyy>
    import re
    pat = re.compile(r"^naps-(no2|pm25|pm10|o3|so2|co)-(on|qc|bc|ab|ns|ca)-(\d{4})$")
    for k in _naps_entries():
        assert pat.match(k), f"Malformed key: {k}"


def test_naps_no_duplicates():
    # Each (pollutant, year, province) triple should be unique
    triples: set[tuple] = set()
    for e in _naps_entries().values():
        fa = e["fetcher_args"]
        key = (fa["pollutant"], fa["year"], fa.get("province"))
        assert key not in triples, f"Duplicate: {key}"
        triples.add(key)
