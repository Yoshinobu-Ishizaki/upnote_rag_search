import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_auto_preprocess
from src.preprocess_runner import maybe_show_dialog, start_preprocess

st.set_page_config(page_title="UpNote RAG Search", layout="wide")

maybe_show_dialog()

if (get_auto_preprocess()
        and not st.session_state.get("_preprocess_done")
        and not st.session_state.get("_pp_show_dialog")):
    st.session_state["_preprocess_done"] = True
    start_preprocess(PROJECT_ROOT)
    st.rerun()

pg = st.navigation([
    st.Page("pages/rag_search.py", title="Search"),
    st.Page("pages/1_Settings.py", title="Settings"),
    st.Page("pages/2_Help.py", title="Help"),
])
pg.run()
