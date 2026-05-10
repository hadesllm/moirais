import importlib
from importlib.metadata import version as _pkg_version


def test_package_import_exposes_stable_top_level_api():
    moirais = importlib.import_module("moirais")

    # __version__ must agree with the installed distribution metadata
    # (the source of truth is pyproject.toml; the test stays in sync
    # automatically across releases instead of pinning a string).
    assert moirais.__version__ == _pkg_version("moirais")
    assert hasattr(moirais, "DatasetRegistry")
    assert hasattr(moirais, "estimate_ate")
    assert not hasattr(moirais, "estimate_double_ml")
