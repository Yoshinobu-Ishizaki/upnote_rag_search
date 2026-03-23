"""Settings page: backup folder path and Anthropic API key."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_api_key, get_backup_path, save_api_key, save_backup_path

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
# Preprocessing instructions
# ---------------------------------------------------------------------------

st.header("前処理の実行")
st.info(
    """設定変更後は、以下のコマンドで前処理を実行してください。

```
python preprocess.py
```

バックアップが大量にある場合（25,000件超）は数時間かかることがあります。
前処理完了後にアプリを再起動すると新しいデータが反映されます。"""
)
