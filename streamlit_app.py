"""PHASE 2.1c bisect entrypoint -- hello-world + Phase 2.1b's 5-dep requirements.

Phase 1 (worked):   1-line requirements + 5-line hello-world.
Phase 2.1  (broke): 6-line requirements + 100-line probe with set_page_config etc.
Phase 2.1b (broke): same 6-line requirements + same 100-line probe (only unpinned streamlit).
Phase 2.1c (now):   same 6-line requirements + revert to 5-line hello-world.

If Phase 2.1c renders "Hello (Phase 2.1c)" -> the deps install cleanly and
my Phase 2.1 entrypoint had the bug (most likely set_page_config / page_icon
/ unicode dashes). Grow the entrypoint back one feature at a time.

If Phase 2.1c still shows "Oh no" -> one of the 5 deps breaks pip-install.
Drop scikit-learn first (heaviest native compile), then numpy, then plotly.
"""
import streamlit as st

st.title("Hello (Phase 2.1c)")
st.write("If you can read this, the 5-dep requirements install cleanly on Cloud.")
st.write("That means my Phase 2.1 entrypoint code was the bug -- not the deps.")
st.success("Phase 2.1c OK -- bug is in my entrypoint, not requirements.txt")
