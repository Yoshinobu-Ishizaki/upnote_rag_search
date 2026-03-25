import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_auto_preprocess

st.set_page_config(page_title="UpNote RAG Search", layout="wide")

if get_auto_preprocess() and not st.session_state.get("_preprocess_done"):
    st.session_state["_preprocess_done"] = True
    with st.spinner("起動時前処理を実行中..."):
        result = subprocess.run(
            ["uv", "run", "python", "preprocess.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.toast("前処理が完了しました。", icon="✅")
    else:
        st.toast("前処理中にエラーが発生しました。Settings ページで詳細を確認してください。", icon="❌")

pg = st.navigation([
    st.Page("pages/rag_search.py", title="Search"),
    st.Page("pages/1_Settings.py", title="Settings"),
    st.Page("pages/2_Help.py", title="Help"),
])
pg.run()
