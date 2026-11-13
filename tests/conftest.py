import sys
import types
from pathlib import Path

# Defensive sys.path shim: ensure the repository root (the parent of the
# `tests/` directory) is on sys.path so `from src.* import ...` works no
# matter how pytest is invoked. This mirrors the `pythonpath = .` setting in
# pytest.ini and makes the suite robust against invocation-form changes.
#
# Keep this ABOVE any project imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# yfinance mock — do not remove; it prevents live network calls during tests.
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
