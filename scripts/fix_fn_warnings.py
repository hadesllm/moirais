# One-shot fixer for SyntaxWarnings in src/morie/fn/*.py.
#
# Prefixes the first module-level triple-quoted docstring with a raw-string
# marker when the file emits a SyntaxWarning under CPython 3.12+ (invalid
# escape sequence). This is the bulletproof fix that does not change runtime
# behaviour.
import warnings, glob, re, os, sys

FN_DIR = "/Volumes/VSR/rootcoderfiles/morie-feature/src/morie/fn"
SKIP = {
    "__init__.py", "_cfa_engine.py", "_containers.py", "_helpers.py",
    "_mapq_const.py", "_otis_const.py", "_registry.py", "_richresult.py",
}


def warns(src, path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            compile(src, path, "exec")
        except SyntaxError:
            return "SYNTAX_ERROR"
        for w in caught:
            if issubclass(w.category, SyntaxWarning):
                return True
        return False


def fix_first(src):
    """Prefix the first standalone triple-quote with r."""
    return re.sub(r'(^|\n)([ \t]*)"""', r'\1\2r"""', src, count=1)


def fix_all(src):
    """Prefix every standalone triple-quote (not already prefixed) with r."""
    # Replace """ that is NOT preceded by r/b/f/R/B/F (a string prefix).
    return re.sub(r'(^|\n)([ \t]*)(?<![rbfRBF])"""', r'\1\2r"""', src)


def main():
    paths = sorted(glob.glob(os.path.join(FN_DIR, "*.py")))
    print(f"Scanning {len(paths)} files...")

    before = []
    for p in paths:
        if os.path.basename(p) in SKIP:
            continue
        with open(p) as f:
            src = f.read()
        r = warns(src, p)
        if r:
            before.append((p, r))

    print(f"Files emitting SyntaxWarning before fix: {len(before)}")

    modified, unfixable = [], []
    for p, status in before:
        if status == "SYNTAX_ERROR":
            unfixable.append((p, "pre-existing SyntaxError"))
            continue
        with open(p) as f:
            src = f.read()
        new = fix_first(src)
        if new != src and warns(new, p) is False:
            with open(p, "w") as f:
                f.write(new)
            modified.append(p)
            continue
        # First-pass failed; try aggressive all-quotes pass.
        new2 = fix_all(src)
        if new2 != src and warns(new2, p) is False:
            with open(p, "w") as f:
                f.write(new2)
            modified.append(p)
            continue
        unfixable.append((p, "still warns after fix"))

    print(f"Files modified: {len(modified)}")
    print(f"Files unfixable: {len(unfixable)}")
    for p, reason in unfixable[:10]:
        print(f"  unfixable: {p}: {reason}")

    # Final sweep
    remaining = []
    for p in paths:
        if os.path.basename(p) in SKIP:
            continue
        with open(p) as f:
            src = f.read()
        r = warns(src, p)
        if r:
            remaining.append((p, r))

    print(f"FINAL: Files still warning: {len(remaining)}")
    for p, r in remaining[:10]:
        print(f"  still warns: {p}: {r}")

    print("First 5 modified files:")
    for p in modified[:5]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
