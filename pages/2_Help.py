import streamlit as st

st.title("UpNote RAG Search")

st.markdown("""
## 使い方

このアプリはUpNoteのバックアップ(.upnxファイル)を検索・照会するツールです。

---

### 初回セットアップ

**1. バックアップパスと API キーを設定**

左サイドバーから **1_Settings** ページを開き、設定を保存してください。

**2. 前処理スクリプトを実行**

```
python preprocess.py
```

> ノート数が多い場合（25,000件超）は数時間かかることがあります。

**3. アプリを使う**

前処理が完了したら、各ページで検索できます。

---

### 前処理の再実行

バックアップを更新した場合は再度実行してください:

```
python preprocess.py
```

特定のフォルダを指定する場合:

```
python preprocess.py --path "D:/backup/UpNote/General Space"
```

---

### ページ一覧

| ページ | 機能 |
|--------|------|
| **RAG Search** | ハイブリッド検索 + Claude AI 回答生成 |
| **1_Settings** | フォルダパス・API キー設定 |
""")
