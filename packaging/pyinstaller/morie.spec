# PyInstaller spec for the standalone morie CLI bundle.
#
# Build:
#   pyinstaller packaging/pyinstaller/morie.spec --noconfirm
#
# Output:
#   dist/morie/            a self-contained directory
#   dist/morie/morie(.exe) the launcher; needs no system Python
#
# The per-OS click-through installers (Windows .exe, macOS .pkg,
# Linux .deb/.rpm) wrap this dist/morie/ directory.

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# SPECPATH is injected by PyInstaller; resolve the launcher next to this spec.
launcher = os.path.join(SPECPATH, "morie_launcher.py")

# morie exposes large parts of its API lazily: a PEP 562 __getattr__ in the
# top-level package and a JSON-backed lazy map in morie.fn. PyInstaller's
# static import analysis cannot see those, so every submodule and every
# non-Python data file is collected explicitly.
hidden = collect_submodules("morie")
datas = collect_data_files("morie")
binaries = collect_dynamic_libs("morie")

# Heavy scientific dependencies. PyInstaller ships hooks for numpy / scipy /
# scikit-learn / matplotlib / pandas, but morie also pulls statsmodels and
# DoubleML, whose submodules are partly imported dynamically. Over-collecting
# here is harmless for a desktop bundle.
for pkg in ("statsmodels", "doubleml"):
    hidden += collect_submodules(pkg)

a = Analysis(
    [launcher],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The optional extras (morie[fast], [interactive], [carbon], [ml]) are
    # deliberately not bundled. morie degrades gracefully when they are
    # absent: numba -> pure-numpy kernels, codecarbon -> emissions off,
    # textual -> the interactive TUI is simply unavailable.
    excludes=[
        "numba",
        "llvmlite",
        "textual",
        "dashing",
        "enlighten",
        "codecarbon",
        "imblearn",
        "imbalanced_learn",
        "pytest",
        "pytest_cov",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="morie",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # morie is a command-line tool
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="morie",
)
