"""Settings page: backup folder path and Anthropic API key."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_api_key, get_auto_preprocess, get_backup_path, get_gemini_api_key, save_api_key, save_auto_preprocess, save_backup_path, save_gemini_api_key
from src.preprocess_runner import start_preprocess

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
# Gemini API key
# ---------------------------------------------------------------------------

st.header("Gemini API キー")
st.caption("Google 埋め込みを使う場合に必要です。`.env` ファイルに保存されます（git管理外）。")

current_gemini_key = get_gemini_api_key()
gemini_placeholder = "AIza..." if not current_gemini_key else "（設定済み）"
new_gemini_key = st.text_input(
    "GEMINI_API_KEY",
    value="",
    placeholder=gemini_placeholder,
    type="password",
)

if st.button("Gemini API キーを保存", key="save_gemini_key"):
    key_to_save = new_gemini_key.strip() if new_gemini_key.strip() else current_gemini_key
    save_gemini_api_key(key_to_save)
    st.success("Gemini API キーを保存しました。")

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
    start_preprocess(PROJECT_ROOT)
    st.rerun()

if st.button("データを再読み込み", key="reload_data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("キャッシュをクリアしました。次の検索で新しいデータが使用されます。")
