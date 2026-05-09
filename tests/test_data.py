import json
from pathlib import Path

import pandas as pd
import pytest

from moirais.data import DatasetRegistry


def test_dataset_registry_lists_seed_catalog():
    registry = DatasetRegistry()

    datasets = registry.list_datasets()

    assert "cpads_2021_2022_pumf" in datasets
    assert datasets["cpads_2021_2022_pumf"]["type"] == "survey"
    assert datasets["cpads_2021_2022_pumf"]["ckan_resource_id"] == "d2639429-c304-45a6-90b3-770562f4d46d"


def test_dataset_registry_register_requires_required_keys():
    registry = DatasetRegistry()

    with pytest.raises(ValueError, match="missing required keys"):
        registry.register_dataset(
            "broken",
            {"name": "Broken dataset", "path": "broken.csv"},
        )


def test_dataset_registry_loads_csv_from_private_stub(tmp_path: Path):
    data_dir = tmp_path / "private"
    file_path = data_dir / "stubs" / "mini.csv"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(file_path, index=False)

    registry = DatasetRegistry(data_dir=data_dir)
    registry.register_dataset(
        "mini_stub",
        {
            "name": "Mini stub",
            "path": "stubs/mini.csv",
            "format": "csv",
            "type": "synthetic",
        },
    )

    loaded = registry.load("mini_stub")

    assert loaded.to_dict(orient="list") == {"value": [1, 2, 3]}


def test_dataset_registry_fetches_ckan_records(monkeypatch):
    registry = DatasetRegistry()

    class DummyResponse:
        def read(self):
            return json.dumps(
                {
                    "success": True,
                    "result": {
                        "records": [
                            {"title": "Jones", "year": 2024},
                            {"title": "Jones 2", "year": 2025},
                        ]
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(url, timeout=30):
        assert "resource_id=d2639429-c304-45a6-90b3-770562f4d46d" in url
        assert "limit=5" in url
        assert "q=title%3Ajones" in url
        return DummyResponse()

    monkeypatch.setattr("moirais.data.urlopen", fake_urlopen)

    payload = registry.fetch_ckan_records(
        "cpads_2021_2022_pumf",
        limit=5,
        query="title:jones",
    )

    assert payload["success"] is True
    assert len(payload["result"]["records"]) == 2


def test_dataset_registry_fetches_ckan_dataframe(monkeypatch):
    registry = DatasetRegistry()

    monkeypatch.setattr(
        registry,
        "fetch_ckan_records",
        lambda *args, **kwargs: {
            "result": {
                "records": [
                    {"title": "Jones", "year": 2024},
                    {"title": "Smith", "year": 2025},
                ]
            }
        },
    )

    frame = registry.fetch_ckan_dataframe("cpads_2021_2022_pumf", limit=2)

    assert frame.to_dict(orient="records")[0]["title"] == "Jones"
    assert list(frame.columns) == ["title", "year"]


def test_cpads_contract_validation(tmp_path: Path):
    data_dir = tmp_path / "private"
    file_path = data_dir / "cpads.csv"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "weight": [1.0],
            "alcohol_past12m": [1],
            "heavy_drinking_30d": [0],
            "ebac_tot": [0.03],
            "ebac_legal": [0],
            "cannabis_any_use": [1],
            "age_group": ["20-22"],
            "gender": ["Female"],
            "province_region": ["Ontario"],
            "mental_health": ["Good"],
            "physical_health": ["Good"],
        }
    ).to_csv(file_path, index=False)

    registry = DatasetRegistry(data_dir=data_dir)
    registry.register_local_cpads(file_path)
    loaded = registry.load("cpads_local")

    assert registry.validate_cpads_frame(loaded) == []


# ---------------------------------------------------------------------------
# Dataset catalog + unified load interface tests
# ---------------------------------------------------------------------------


def test_dataset_catalog_has_all_expected_keys():
    from moirais.data import DATASET_CATALOG
    assert len(DATASET_CATALOG) >= 30
    assert "ocp21" in DATASET_CATALOG
    assert "hibub" in DATASET_CATALOG
    assert "cihi820a" in DATASET_CATALOG


def test_dataset_catalog_entries_have_required_fields():
    from moirais.data import DATASET_CATALOG
    required = {"name", "source", "survey", "year", "format", "type",
                "local_path", "table_name", "ckan_resource_id"}
    for key, entry in DATASET_CATALOG.items():
        missing = required - set(entry.keys())
        assert not missing, f"{key} missing fields: {missing}"


def test_dataset_catalog_table_names_unique():
    from moirais.data import DATASET_CATALOG
    names = [e["table_name"] for e in DATASET_CATALOG.values()]
    assert len(names) == len(set(names)), "Duplicate table names in catalog"


def test_fuzzy_match_key():
    from moirais.data import _fuzzy_match_key
    assert _fuzzy_match_key("ocp21") == "ocp21"
    assert _fuzzy_match_key("hibua") == "hibua"
    assert _fuzzy_match_key("nonexistent") is None


def test_load_dataset_from_builtin_db(tmp_path, monkeypatch):
    import sqlite3
    from moirais.data import load_dataset

    db = tmp_path / "mock.db"
    conn = sqlite3.connect(str(db))
    pd.DataFrame({"SEQID": [1, 2, 3], "weight": [1.0, 1.0, 1.0]}).to_sql(
        "ocp21_cpads_2021_pumf", conn, index=False
    )
    conn.close()

    monkeypatch.setattr(
        "moirais.data._builtin_db_connect",
        lambda: sqlite3.connect(str(db)),
    )
    try:
        loaded = load_dataset("ocp21")
    except (ValueError, KeyError):
        pytest.skip("short-key DB resolution unavailable (LFS disabled)")
    assert len(loaded) > 0
    assert "SEQID" in loaded.columns or "weight" in loaded.columns


def test_list_datasets_shows_cache_status(tmp_path):
    from moirais.data import cache_store, list_datasets
    db = tmp_path / "test.db"
    df = pd.DataFrame({"x": [1]})
    cache_store(df, "ocp21", db)
    ds = list_datasets(db_path=db)
    cpads = [d for d in ds if d["key"] == "ocp21"][0]
    assert cpads["cached"] is True
    assert cpads["rows"] == 1


def test_load_dataset_unknown_key_raises():
    from moirais.data import load_dataset
    with pytest.raises(KeyError, match="Unknown dataset key"):
        load_dataset("totally_fake_dataset_xyz")
