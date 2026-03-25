"""Settings page: backup folder path and Anthropic API key."""
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_api_key, get_auto_preprocess, get_backup_path, save_api_key, save_auto_preprocess, save_backup_path

st.title("Settings")

# ---------------------------------------------------------------------------
# Backup folder path
# ---------------------------------------------------------------------------

st.header("バックアップフォルダ")
st.caption("UpNote のバックアップ先フォルダを指定してください。")

current_path = str(get_backup_path())
new_path = st.text_input("バックアップフォルダのパス", value=current_path)

if st.button("パスを保存", key="save_path"):
    p = Path(new_path)
    if not p.exists():
        st.warning(f"パスが存在しません: {new_path}\n設定は保存しますが、前処理時にエラーになる可能性があります。")
    save_backup_path(new_path)
    st.success("バックアップパスを保存しました。")

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

st.header("Anthropic API キー")
st.caption("Claude API の利用に必要です。`.env` ファイルに保存されます（git管理外）。")

current_key = get_api_key()
placeholder = "sk-ant-..." if not current_key else "（設定済み）"
new_key = st.text_input(
    "ANTHROPIC_API_KEY",
    value="",
    placeholder=placeholder,
    type="password",
)

if st.button("API キーを保存", key="save_key"):
    key_to_save = new_key.strip() if new_key.strip() else current_key
    save_api_key(key_to_save)
    st.success("API キーを保存しました。")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

st.header("前処理の実行")
st.caption("バックアップが大量にある場合（25,000件超）は数時間かかることがあります。")

auto = st.toggle("起動時に自動で前処理を実行", value=get_auto_preprocess())
if auto != get_auto_preprocess():
    save_auto_preprocess(auto)
    st.success(f"自動前処理を{'有効' if auto else '無効'}にしました。")

if st.button("前処理を実行", key="run_preprocess"):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        ["uv", "run", "python", "-u", "preprocess.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_area = st.empty()
    lines = []
    for line in process.stdout:
        lines.append(line)
        output_area.code("".join(lines[-50:]))
    process.wait()
    output_area.code("".join(lines))

    if process.returncode == 0:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("前処理が完了しました。キャッシュをクリアしました。次の検索で新しいデータが使用されます。")
    else:
        st.error("前処理中にエラーが発生しました。上記のログを確認してください。")

if st.button("データを再読み込み", key="reload_data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("キャッシュをクリアしました。次の検索で新しいデータが使用されます。")
