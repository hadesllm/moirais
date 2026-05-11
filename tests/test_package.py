import importlib
from importlib.metadata import version as _pkg_version


def test_package_import_exposes_stable_top_level_api():
    morie = importlib.import_module("morie")

    # __version__ must agree with the installed distribution metadata
    # (the source of truth is pyproject.toml; the test stays in sync
    # automatically across releases instead of pinning a string).
    assert morie.__version__ == _pkg_version("morie")
    assert hasattr(morie, "DatasetRegistry")
    assert hasattr(morie, "estimate_ate")
    assert not hasattr(morie, "estimate_double_ml")
