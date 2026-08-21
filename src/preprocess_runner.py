"""Shared preprocess dialog module used by app.py and Settings page."""
import os
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st

_HIDE_CLOSE_CSS = """<style>
[data-baseweb="modal"] button[aria-label="Close"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}
</style>"""


def _init_state() -> None:
    if "_pp_show_dialog" not in st.session_state:
        st.session_state["_pp_show_dialog"] = False
    if "_pp_process" not in st.session_state:
        st.session_state["_pp_process"] = None
    # _pp_lines and _pp_holder are set fresh in start_preprocess


def start_preprocess(project_root: Path) -> None:
    _init_state()
    holder = st.session_state.get("_pp_holder")
    if holder and holder["state"] == "running":
        return

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        ["uv", "run", "python", "-u", "preprocess.py"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Mutable objects shared with the thread — thread NEVER touches st.session_state
    lines: list[str] = []
    lock = threading.Lock()
    holder = {"state": "running", "returncode": None}

    st.session_state["_pp_holder"] = holder
    st.session_state["_pp_lines"] = lines
    st.session_state["_pp_lock"] = lock
    st.session_state["_pp_process"] = process
    st.session_state["_pp_show_dialog"] = True

    t = threading.Thread(
        target=_reader_thread,
        args=(process, lines, lock, holder),
        daemon=True,
    )
    t.start()


def _reader_thread(
    process: subprocess.Popen,
    lines: list,
    lock: threading.Lock,
    holder: dict,
) -> None:
    """Runs in background thread — must NOT access st.session_state."""
    for line in process.stdout:
        with lock:
            lines.append(line)
    process.wait()
    with lock:
        if holder["state"] == "running":
            holder["returncode"] = process.returncode
            holder["state"] = "done" if process.returncode == 0 else "error"


@st.dialog("前処理", width="large")
def preprocess_dialog() -> None:
    st.markdown(_HIDE_CLOSE_CSS, unsafe_allow_html=True)

    holder = st.session_state.get("_pp_holder", {"state": "not_started"})
    lock = st.session_state.get("_pp_lock")
    lines = st.session_state.get("_pp_lines", [])

    # Read state under lock to get consistent snapshot
    with lock:
        state = holder["state"]
        output = "".join(lines[-80:])

    st.code(output, language=None)

    col_status, col_cancel = st.columns([4, 1])

    with col_cancel:
        btn_label = "閉じる" if state in ("error", "cancelled") else "キャンセル"
        if st.button(btn_label, type="secondary", use_container_width=True):
            process = st.session_state.get("_pp_process")
            if process and process.poll() is None:
                process.kill()
            with lock:
                holder["state"] = "cancelled"
            st.session_state["_pp_show_dialog"] = False
            st.rerun()

    with col_status:
        if state == "running":
            st.caption("実行中...")
        elif state == "error":
            st.error("前処理中にエラーが発生しました。上記のログを確認してください。")
        elif state == "done":
            st.success("前処理が完了しました。")

    if state == "done":
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["_preprocess_done"] = True
        st.session_state["_pp_show_dialog"] = False
        time.sleep(1.0)
        st.rerun()

    if state == "running":
        time.sleep(0.5)
        st.rerun()


def maybe_show_dialog() -> None:
    _init_state()
    if st.session_state.get("_pp_show_dialog"):
        preprocess_dialog()
