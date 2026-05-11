"""Tests for morie._migration — verify migration is complete.

The migration from the old epiml-project layout into morie is DONE.
All 21 modules are inlined in morie_pipeline.R and registered in MODULE_SPECS.
"""

from morie._migration import LEGACY_MODULE_DOCS, LEGACY_TO_MODULE


def test_legacy_to_module_map_covers_all_scripts():
    """LEGACY_TO_MODULE should map every script in LEGACY_MODULE_DOCS."""
    for script in LEGACY_MODULE_DOCS:
        assert script in LEGACY_TO_MODULE, (
            f"Legacy script '{script}' has docs mapping but no module mapping"
        )


def test_all_mapped_modules_exist_in_specs():
    """Every module in LEGACY_TO_MODULE should exist in MODULE_SPECS."""
    from morie.modules import MODULE_SPECS
    for script, module_name in LEGACY_TO_MODULE.items():
        assert module_name in MODULE_SPECS, (
            f"Legacy script '{script}' maps to module '{module_name}' "
            f"which is not in MODULE_SPECS"
        )


def test_all_21_modules_mapped():
    """There should be at least 21 legacy-to-module mappings."""
    assert len(LEGACY_TO_MODULE) >= 21, (
        f"Only {len(LEGACY_TO_MODULE)} mappings, expected 21+"
    )
