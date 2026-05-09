"""
REPL bootstrap for TIDE (Go TUI) and any external REPL.

Usage from TIDE subprocess:
    exec("from moirais.repl_init import setup_repl; setup_repl(globals())")

This injects all convenience functions, stat_commands, and fn/ registry
into the caller's namespace — achieving parity with the Python TUI REPL.
"""

from __future__ import annotations


def setup_repl(ns: dict) -> None:
    """Inject all MOIRAIS REPL helpers into namespace `ns`."""
    import numpy as np
    import pandas as pd

    ns["np"] = np
    ns["pd"] = pd

    _inject_convenience(ns)
    _inject_stat_commands(ns)
    _inject_fn_registry(ns)


def _inject_stat_commands(ns: dict) -> None:
    try:
        from moirais.stat_commands import ALIAS_MAP, COMMAND_REGISTRY

        for name, cmd in COMMAND_REGISTRY.items():
            if name not in ns:
                ns[name] = cmd.handler_repl
        for alias, canonical in ALIAS_MAP.items():
            if alias not in ns and canonical in COMMAND_REGISTRY:
                ns[alias] = COMMAND_REGISTRY[canonical].handler_repl
    except Exception:
        pass


def _inject_fn_registry(ns: dict) -> None:
    try:
        from moirais.fn._registry import REGISTRY

        for key in REGISTRY:
            if key not in ns:
                try:
                    mod = __import__(f"moirais.fn.{key}", fromlist=[key])
                    ns[key] = getattr(mod, key)
                except Exception:
                    pass
    except Exception:
        pass


def _inject_convenience(ns: dict) -> None:
    """Define and inject all convenience functions (parity with tui.py ReplScreen)."""

    def load(path_or_name=None):
        """Load dataset. load('ocp21') or load('file.csv')."""
        if path_or_name is None:
            from moirais.data import list_datasets

            print("Usage: load('name') or load('path.csv')")
            print("\n  Available datasets:")
            try:
                for d in list_datasets():
                    tag = f" ({d['rows']} rows)" if d.get("rows") else ""
                    print(f"    load('{d['key']}'){tag}")
            except Exception as e:
                print(f"    (error: {e})")
            return None
        try:
            from moirais.data import load_dataset

            df = load_dataset(path_or_name)
            var = path_or_name.replace("-", "_").replace(" ", "_").lower()
            ns[var] = df
            ns["df"] = df
            print(f"Loaded {path_or_name}: {df.shape[0]} rows x {df.shape[1]} cols -> '{var}' and 'df'")
            return df
        except (KeyError, FileNotFoundError):
            pass
        from pathlib import Path

        import pandas as _pd

        for p in [Path(path_or_name), Path.cwd() / path_or_name]:
            if p.exists():
                df = _pd.read_csv(p, low_memory=False)
                name = p.stem.replace("-", "_")
                ns[name] = df
                ns["df"] = df
                print(f"Loaded {p.name}: {df.shape[0]} rows x {df.shape[1]} cols -> '{name}' and 'df'")
                return df
        print(f"Not found: {path_or_name}. Use load() for available datasets.")
        return None

    def head(obj=None, n=10):
        """First n rows."""
        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("No data. Use load() first.")
            return
        if hasattr(obj, "head"):
            print(obj.head(n).to_string())
        else:
            print(repr(obj)[:500])

    def tail(obj=None, n=10):
        """Last n rows."""
        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("No data.")
            return
        if hasattr(obj, "tail"):
            print(obj.tail(n).to_string())
        else:
            print(repr(obj)[-500:])

    def shape(obj=None):
        """Show shape."""
        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("No data.")
            return
        if hasattr(obj, "shape"):
            print(f"  {obj.shape[0]} rows x {obj.shape[1]} columns")
        else:
            print(f"  type: {type(obj).__name__}")

    def cols(obj=None):
        """List columns and dtypes."""
        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("No data.")
            return
        if hasattr(obj, "dtypes"):
            for c in obj.columns:
                miss = obj[c].isna().sum()
                m = f" ({miss} missing)" if miss > 0 else ""
                print(f"  {c}: {obj[c].dtype}{m}")

    def describe(obj=None):
        """Descriptive statistics."""
        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("No data.")
            return
        if hasattr(obj, "describe"):
            print(obj.describe().to_string())

    def sample(obj=None, n=5):
        """Random n rows."""
        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("No data.")
            return
        s = obj.sample(n=min(n, len(obj)))
        print(s.to_string())
        return s

    def missing(data=None):
        """Missing data report."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        nulls = data.isnull().sum()
        total = len(data)
        has_missing = nulls[nulls > 0]
        if len(has_missing) == 0:
            print("  No missing values.")
        else:
            print(f"  Missing values ({len(has_missing)}/{len(data.columns)} columns):")
            for c, n in has_missing.items():
                print(f"    {c}: {n} ({n / total * 100:.1f}%)")

    def unique(col=None, data=None):
        """Unique values of a column."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        if col is None:
            print("Usage: unique('col')")
            return
        vals = data[col].unique()
        print(f"  {col}: {len(vals)} unique values")
        show = sorted(vals, key=str)[:20]
        for v in show:
            print(f"    {v}")
        if len(vals) > 20:
            print(f"    ... ({len(vals) - 20} more)")

    def value_counts(col=None, data=None, n=20):
        """Frequency table."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        if col is None:
            print("Usage: value_counts('col')")
            return
        vc = data[col].value_counts().head(n)
        total = len(data)
        for val, count in vc.items():
            pct = count / total * 100
            print(f"    {str(val):20s} {count:6d} ({pct:5.1f}%)")

    def filter_rows(condition=None, data=None):
        """Filter rows. filter_rows('col > 5')."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        if condition is None:
            print("Usage: filter_rows('condition')")
            return
        if isinstance(condition, str):
            result = data.query(condition)
        else:
            result = data[condition]
        ns["filtered"] = result
        print(f"  Filtered: {len(result)}/{len(data)} rows -> 'filtered'")
        return result

    def select_cols(*columns, data=None):
        """Select columns. select_cols('c1', 'c2')."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        result = data[list(columns)]
        ns["selected"] = result
        print(f"  Selected {len(columns)} cols -> 'selected'")
        return result

    def rename_col(old=None, new=None, data=None):
        """Rename column."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        if old is None or new is None:
            print("Usage: rename_col('old', 'new')")
            return
        data = data.rename(columns={old: new})
        ns["df"] = data
        print(f"  Renamed '{old}' -> '{new}'")
        return data

    def dropna(col=None, data=None):
        """Drop NA rows."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        before = len(data)
        data = data.dropna(subset=[col]) if col else data.dropna()
        ns["df"] = data
        print(f"  Dropped {before - len(data)} rows ({len(data)} remaining)")
        return data

    def crosstab(c1=None, c2=None, data=None):
        """Cross-tabulation."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        if c1 is None or c2 is None:
            print("Usage: crosstab('c1', 'c2')")
            return
        import pandas as _pd

        ct = _pd.crosstab(data[c1], data[c2])
        print(ct.to_string())
        return ct

    def groupby(grp=None, col=None, fn="mean", data=None):
        """Group-by aggregation."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        if grp is None or col is None:
            print("Usage: groupby('grp', 'col', 'mean')")
            return
        result = data.groupby(grp)[col].agg(fn)
        print(result.to_string())
        return result

    def save(path="output.csv", data=None):
        """Save to CSV."""
        if data is None:
            data = ns.get("df")
        if data is None:
            print("No data.")
            return
        data.to_csv(path, index=False)
        print(f"  Saved {len(data)} rows to {path}")

    def view(obj=None):
        """Pretty-print any object."""
        import pprint

        if obj is None:
            obj = ns.get("df")
        if obj is None:
            print("Usage: view(variable)")
            return
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            pprint.pprint(obj)

    def ls():
        """List user variables."""
        skip = {"__name__", "__builtins__", "__doc__", "np", "pd", "moirais"}
        user = {
            k: type(v).__name__ for k, v in ns.items() if not k.startswith("_") and k not in skip and not callable(v)
        }
        if user:
            for name, typ in sorted(user.items()):
                print(f"  {name}: {typ}")
        else:
            print("  (no user variables)")

    def who():
        """Variables with types and values."""
        skip = {"__name__", "__builtins__", "__doc__", "np", "pd", "moirais"}
        user = {k: v for k, v in ns.items() if not k.startswith("_") and k not in skip and not callable(v)}
        if user:
            for name, val in sorted(user.items()):
                rep = repr(val)
                if len(rep) > 60:
                    rep = rep[:57] + "..."
                print(f"  {name} ({type(val).__name__}): {rep}")
        else:
            print("  (no user variables)")

    def clear():
        """Clear user variables."""
        skip = {"__name__", "__builtins__", "__doc__", "np", "pd", "moirais"}
        to_rm = [k for k in ns if not k.startswith("_") and k not in skip and not callable(ns[k])]
        for k in to_rm:
            del ns[k]
        print(f"Cleared {len(to_rm)} variable(s)")

    def modules():
        """List MOIRAIS modules."""
        from moirais.modules import list_modules

        for m in list_modules():
            print(f"  {m['name']}: {m['description']}")

    def run_module(name=None, **kwargs):
        """Run an MOIRAIS module."""
        if name is None:
            print("Usage: run_module('name')")
            modules()
            return
        from moirais.modules import run_module as _rm

        print(f"Running {name}...")
        return _rm(name, **kwargs)

    def selftest():
        """Run MOIRAIS self-test."""
        from moirais.selftest import run_selftest

        for r in run_selftest():
            s = "PASS" if r.get("ok") else "FAIL"
            print(f"  {s}  {r.get('name', '?')}")

    def doctor():
        """Run environment diagnostics."""
        from moirais.doctor import run_doctor

        run_doctor()

    def version():
        """Show version info."""
        import moirais

        print(f"  moirais {getattr(moirais, '__version__', 'dev')}")

    def help_repl():
        """Show all REPL helper functions."""
        sections = {
            "Data": "load head tail shape cols describe sample missing save",
            "Wrangling": "filter_rows select_cols rename_col dropna crosstab groupby",
            "Inspect": "unique value_counts view ls who clear",
            "System": "modules run_module selftest doctor version help_repl",
        }
        print("  MOIRAIS REPL Helpers:")
        for section, fns in sections.items():
            print(f"\n  [{section}]")
            for fn in fns.split():
                print(f"    {fn}()")
        print("\n  + 1130 fn/ functions (dnorm, ttest, ate, force, luke ...)")
        print("  + 620 stat_commands (all callable directly)")

    helpers = {
        "load": load,
        "head": head,
        "tail": tail,
        "shape": shape,
        "cols": cols,
        "describe": describe,
        "sample": sample,
        "missing": missing,
        "unique": unique,
        "value_counts": value_counts,
        "filter_rows": filter_rows,
        "select_cols": select_cols,
        "rename_col": rename_col,
        "dropna": dropna,
        "crosstab": crosstab,
        "groupby": groupby,
        "save": save,
        "view": view,
        "ls": ls,
        "who": who,
        "clear": clear,
        "modules": modules,
        "run_module": run_module,
        "selftest": selftest,
        "doctor": doctor,
        "version": version,
        "help_repl": help_repl,
    }

    for name, fn in helpers.items():
        ns[name] = fn
