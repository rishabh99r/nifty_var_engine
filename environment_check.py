# environment_check.py
# =============================================================================
# Dependency sanity check (S6, reproducibility).
#
# Asserts that the runtime environment matches the validated research matrix
# (see requirements.txt / requirements.lock) BEFORE a long training run starts,
# so a silent package-version drift cannot invalidate the empirical results.
#
# Usage:  python environment_check.py
# =============================================================================
import importlib
import platform
import sys


def _ver(mod):
    try:
        m = importlib.import_module(mod)
    except ImportError:
        return None
    return getattr(m, "__version__", "unknown")


def _check(name, pkg, expected):
    got = _ver(pkg)
    ok = got == expected
    status = "OK " if ok else "MISMATCH"
    print(f"[{status}] {name:26s} expected={expected:14s} got={got}")
    return ok


def main():
    print("=" * 74)
    print("ENVIRONMENT CHECK  (research reproducibility gate)")
    print(f"Python: {sys.version.split()[0]}  (interpreter: {platform.python_implementation()})")
    print("=" * 74)

    ok = True
    # Expected versions from the pinned requirements matrix.
    checks = [
        ("torch", "torch", "2.2.2"),
        ("pytorch-forecasting", "pytorch_forecasting", "1.0.0"),
        ("lightning", "lightning", "2.2.5"),
        ("arch", "arch", "7.0.0"),
        ("pandas", "pandas", "2.2.2"),
        ("numpy", "numpy", "1.26.4"),
        ("scipy", "scipy", "1.13.1"),
    ]
    for name, pkg, expected in checks:
        ok = _check(name, pkg, expected) and ok

    # Hard fail if any core import is missing.
    missing = [pkg for _, pkg, _ in checks if _ver(pkg) is None]
    if missing:
        print("\n[FATAL] Missing packages:", ", ".join(missing))
        print("        Run:  pip install -r requirements.txt")
        sys.exit(1)

    if not ok:
        print("\n[WARNING] Version mismatches detected. Freeze the environment with:")
        print("          pip freeze > requirements.lock")
        print("          before treating results as the reproducible final run.")
        sys.exit(1)

    print("\n[OK] Environment matches the validated research matrix.")


if __name__ == "__main__":
    main()
