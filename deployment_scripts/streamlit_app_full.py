"""Streamlit Cloud entrypoint shim.

Streamlit Cloud expects a single Python file at the repo root (or a
manifest pointing at one). The actual dashboard lives at
``dashboards/churn_dashboard_app.py``; this file simply re-executes it
so we don't duplicate code.

Local dev still runs ``streamlit run dashboards/churn_dashboard_app.py``
directly. Both paths read the same predictions CSV + model joblib.

Implementation note: we use ``exec(compile(...))`` rather than
``runpy.run_path`` so the dashboard code runs in THIS module's globals
(``__name__ == "__main__"``). ``runpy`` would create a fresh module
namespace which confuses Streamlit's session state and decorator caches
on Streamlit Cloud (manifested as a redacted "Oh no" page with no
recoverable traceback).
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Make `src` (and the project root) importable inside the Streamlit Cloud sandbox.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("CHURN_BASE_DIR", str(_ROOT))


# ---------------------------------------------------------------------------
# Pre-import diagnostic — Streamlit Cloud redacts the actual ModuleNotFoundError
# message when shown in the browser; we surface it to stdout (visible in the
# "Manage app" panel) AND to the dashboard itself so the next failure is
# self-explanatory.
# ---------------------------------------------------------------------------
def _probe_imports() -> list[str]:
    """Try the heavy imports the dashboard does and capture any failures."""
    failures: list[str] = []
    for mod in [
        "streamlit",
        "pandas",
        "numpy",
        "joblib",
        "plotly.express",
        "plotly.graph_objects",
        "sklearn.metrics",
        "lightgbm",
        "category_encoders",
        "imblearn",
        "dill",
        "loky",
        "threadpoolctl",
        "cloudpickle",
    ]:
        try:
            __import__(mod)
            print(f"[probe] OK {mod}")
        except Exception as exc:
            msg = f"{mod}: {type(exc).__name__}: {exc}"
            print(f"[probe] FAIL {msg}")
            failures.append(msg)
    return failures


_failures = _probe_imports()

if _failures:
    # Render a Streamlit error page that is NOT redacted — we control the
    # text so Streamlit Cloud will show it verbatim.
    try:
        import streamlit as st
        st.set_page_config(page_title="Telecom Churn — import error",
                           page_icon="⚠")
        st.error("Some dependencies failed to import. Full traceback below.")
        st.code("\n".join(_failures), language="text")
        st.caption(
            "If you're on Streamlit Cloud: this means `requirements.txt` is "
            "missing one of these modules. Open `requirements.txt`, add the "
            "missing name (with a compatible version pin), commit and push. "
            "Streamlit Cloud auto-rebuilds in ~30s."
        )
        st.stop()
    except Exception:
        # Last resort: print and re-raise so the redacted error path fires.
        traceback.print_exc()
        raise ImportError("\n".join(_failures))


# Execute the dashboard code IN PLACE so Streamlit sees this file as
# `__main__` (matching what `streamlit run streamlit_app.py` expects).
_DASH = _ROOT / "dashboards" / "churn_dashboard_app.py"
try:
    _source = _DASH.read_text(encoding="utf-8")
    _code = compile(_source, str(_DASH), "exec")
    # Run in our globals — preserves __name__ == "__main__" and lets the
    # dashboard's @st.cache_data / @st.cache_resource decorators see the
    # same session Streamlit Cloud handed us.
    _globals = globals().copy()
    _globals["__file__"] = str(_DASH)
    exec(_code, _globals)
except Exception:
    # Last-ditch: render the traceback in Streamlit so the next debug
    # iteration is targeted (instead of the redacted "Oh no" page).
    try:
        import streamlit as st
        st.set_page_config(page_title="Telecom Churn — dashboard error",
                           page_icon="⚠", layout="wide")
        st.error("Dashboard code raised an exception during execution.")
        st.code(traceback.format_exc(), language="text")
        st.stop()
    except Exception:
        traceback.print_exc()
        raise
