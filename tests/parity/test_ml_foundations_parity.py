"""Smoke harness for ML Foundations callables.

Loads each fn/<name>.py directly to bypass morie.fn.__init__ side-effects.
Run as:

    python3 _smoke_ml.py [name1 name2 ...]
"""
import sys
import os
import importlib.util
import types
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

pkg = types.ModuleType("morie")
pkg.__path__ = [os.path.join(SRC, "morie")]
sys.modules["morie"] = pkg
fnpkg = types.ModuleType("morie.fn")
fnpkg.__path__ = [os.path.join(SRC, "morie", "fn")]
sys.modules["morie.fn"] = fnpkg


def _load(modname, relpath):
    full = f"morie.fn.{modname}" if modname != "_richresult" else "morie.fn._richresult"
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(SRC, "morie", "fn", relpath)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


_load("_richresult", "_richresult.py")


def run_one(name):
    print(f"=== {name} ===")
    try:
        m = _load(name, f"{name}.py")
        src = open(os.path.join(SRC, "morie", "fn", f"{name}.py")).read()
        if 'if __name__ == "__main__":' in src:
            tail = src.split('if __name__ == "__main__":', 1)[1]
            lines = tail.splitlines()
            body = "\n".join(L[4:] if L.startswith("    ") else L for L in lines)
            ns = {"__name__": "__smoke__"}
            ns.update(m.__dict__)
            exec(body, ns)
        print(f"OK: {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {name}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "linrg", "grdds", "mbgrd", "polrg", "rgztn", "lrcvg",
        "svmhg", "svmkr", "dtrsp", "rfens", "gbens", "xgbst",
        "pcadm", "tsnrd", "kmnsc", "dbscl", "gsrch", "rndsr",
        "confm", "rocau",
    ]
    results = {n: run_one(n) for n in names}
    print()
    print("Summary:")
    for n, ok in results.items():
        print(f"  {n}: {'PASS' if ok else 'FAIL'}")
