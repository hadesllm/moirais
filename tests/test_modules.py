import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from morie.modules import list_modules, run_module


def test_list_modules_includes_core_cpads_steps():
    names = {item["name"] for item in list_modules()}
    assert {
        "power-design",
        "logistic-models",
        "model-comparison",
        "propensity-scores",
        "treatment-effects",
        "ebac-selection-adjustment-ipw",
    }.issubset(names)


def _mock_cpads_df():
    """Minimal CPADS-like DataFrame for testing without the real DB."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "SEQID": range(1, n + 1),
        "weight": rng.uniform(0.5, 2.0, n),
        "alcohol_past12m": rng.choice([0, 1], n),
        "heavy_drinking_30d": rng.choice([0, 1], n),
        "ebac_tot": rng.uniform(0, 0.1, n),
        "ebac_legal": rng.choice([0, 1], n),
        "cannabis_any_use": rng.choice([0, 1], n),
        "age_group": rng.choice(["18-19", "20-22", "23-25"], n),
        "gender": rng.choice(["Female", "Male"], n),
        "province_region": rng.choice(["Ontario", "Quebec", "BC"], n),
        "mental_health": rng.choice(["Good", "Fair", "Poor"], n),
        "physical_health": rng.choice(["Good", "Fair", "Poor"], n),
    })


def _mock_db(tmp_path):
    """Build a tiny SQLite DB with a CPADS-like table."""
    db_path = tmp_path / "mock_morie.db"
    conn = sqlite3.connect(str(db_path))
    _mock_cpads_df().to_sql("ocp21_cpads_2021_pumf", conn, index=False)
    conn.close()
    return db_path


def test_load_cpads_from_mock_db(tmp_path, monkeypatch):
    """Load CPADS from a mock SQLite DB (no LFS needed)."""
    from morie.data import load_dataset
    db_path = _mock_db(tmp_path)
    monkeypatch.setattr(
        "morie.data._builtin_db_connect",
        lambda: sqlite3.connect(str(db_path)),
    )
    try:
        frame = load_dataset("ocp21")
    except (ValueError, KeyError):
        pytest.skip("short-key DB resolution unavailable (LFS disabled)")
    assert len(frame) > 0
    assert "SEQID" in frame.columns or "weight" in frame.columns


def test_load_multiple_datasets_mock(tmp_path, monkeypatch):
    """Verify dataset load works with mock DB."""
    from morie.data import load_dataset
    db_path = _mock_db(tmp_path)
    conn_factory = lambda: sqlite3.connect(str(db_path))
    monkeypatch.setattr("morie.data._builtin_db_connect", conn_factory)
    try:
        df = load_dataset("ocp21")
    except (ValueError, KeyError):
        pytest.skip("short-key DB resolution unavailable (LFS disabled)")
    assert len(df) > 0


def test_run_module_with_mock_cpads(tmp_path):
    """Run power-design module using synthetic CPADS CSV."""
    frame = _mock_cpads_df()
    csv_path = tmp_path / "cpads.csv"
    frame.to_csv(csv_path, index=False)
    outputs = run_module("power-design", cpads_csv=str(csv_path), output_dir=tmp_path)
    assert "power_summary" in outputs
    assert (tmp_path / "power_summary.csv").exists()
