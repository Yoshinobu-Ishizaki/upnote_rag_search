import streamlit as st

st.title("UpNote RAG Search")

st.markdown("""
## 使い方

このアプリはUpNoteのバックアップ(.upnxファイル)を検索・照会するツールです。

---

### 初回セットアップ

**1. バックアップパスと API キーを設定**

`config.sample.ini` を `config.ini` にコピーして編集するか、
左サイドバーから **Settings** ページを開いて設定を保存してください。

APIキーは `.env` ファイルに記載します（gitで管理されません）:

```
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here   # Google埋め込みを使う場合のみ
```

**2. 前処理スクリプトを実行**

```
uv run python preprocess.py
```

> ノート数が多い場合（25,000件超）は数時間かかることがあります。
> 中断しても次回実行時にチェックポイントから再開します。

特定のフォルダを指定する場合:

```
uv run python preprocess.py --path "D:/backup/UpNote/General Space"
```

埋め込み・FAISSインデックスのみ再構築する場合（ステップ1・2をスキップ）:

```
uv run python preprocess.py --embeddings-only
```

**Settings** ページのボタンからアプリ内で実行することもできます。

**3. アプリを使う**

前処理が完了したら、**RAG Search** ページで検索できます。

---

### Settings ページ

| 項目 | 説明 |
|------|------|
| バックアップパス | UpNote バックアップディレクトリのパス |
| Anthropic API キー | Claude API キー |
| 自動前処理 | アプリ起動時に前処理を自動実行するか切り替え |
| 前処理を実行 | 前処理をその場で実行（リアルタイム出力表示） |
| データを再読み込み | キャッシュをクリアしてデータを再読み込み |

---

### 検索フィルター

サイドバーで以下のフィルターを設定できます。

| フィルター | 説明 |
|-----------|------|
| カテゴリ | ノートのカテゴリで絞り込み（複数選択可） |
| タグ | タグで絞り込み（複数選択可） |
| 作成日 | 日付条件: すべて / 以前 / 以降 / 範囲 |
| 参照ノート数 | 取得するノートの最大件数 |

フィルターの初期値は `config.ini` で設定できます:

```ini
DEFAULT_DATE_MODE = 以降
DEFAULT_CATEGORIES = MyCategory
DEFAULT_TAGS =
DEFAULT_START_DATE = 2024-01-01
DEFAULT_END_DATE =
```

---

### 埋め込みプロバイダー

`config.ini` の `EMBEDDING_PROVIDER` で切り替えられます。

| プロバイダー | モデル | 検索方式 |
|------------|--------|---------|
| `local`（デフォルト） | paraphrase-multilingual-mpnet-base-v2（ローカル実行） | BM25 + 意味検索（RRF融合） |
| `google` | gemini-embedding-001（Google API、3072次元） | 意味検索のみ |

`google` を使う場合は `.env` に `GEMINI_API_KEY` を設定してください。
両プロバイダーのインデックスは共存できます（`local` は `data/`、`google` は `data/google/`）。

---

### ページ一覧

| ページ | 機能 |
|--------|------|
| **RAG Search** | ハイブリッド検索 + Claude AI 回答生成 |
| **Settings** | フォルダパス・API キー・前処理設定 |
| **Help** | このページ |
""")
