UpNote RAG Search
===

yoshinobu ishizaki

## Abstract

Hybrid RAG (Retrieval-Augmented Generation) search for UpNote notes.
Combines BM25 keyword matching and semantic vector search (RRF fusion),
then generates answers via Claude API.

## Requirements

- UpNote desktop app (to export `.upnx` backup)
- Anthropic API key
- Python environment managed by uv
- Google Generative AI API key (optional, for Google embeddings)

## Setup

### 1. Prepare backup file

Export your UpNote data as a native backup (`.upnx` file) from the UpNote app.

### 2. Configure settings

Copy `config.sample.ini` to `config.ini` and edit the values.
See `config.sample.ini` for all available keys and their descriptions.

Set API keys in `.env` (not tracked by git):

```
ANTHROPIC_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here   # only needed for Google embeddings
```

Alternatively, configure everything from the **Settings** page in the app UI.

### 3. Preprocess data

Run the preprocessing pipeline to build the search index:

    uv run python preprocess.py

This will:
1. Parse the `.upnx` file → `data/upnote_text.csv`
2. Tokenize with Sudachi → `data/upnote_text_split.csv`
3. Build FAISS semantic index → `data/local/faiss.index`

To rebuild only the embeddings/FAISS index (skipping steps 1 & 2), use:

    uv run python preprocess.py --embeddings-only

This is useful if you interrupted a previous run — the script saves a checkpoint after each batch and resumes automatically.

You can also run preprocessing from the **Settings** page in the app, which streams live output and reloads data automatically on completion.

### 4. Launch the app

    uv run python -m streamlit run app.py

## Usage

### Search

Open the **RAG Search** page, enter a query in natural language, then click **検索・回答生成**.
The app returns a Claude-generated answer with a referenced notes grid showing scores, categories, tags, and content previews.

**Sidebar filters:**

| Filter | Description |
|--------|-------------|
| カテゴリ | Pre-filter by notebook category (multiselect) |
| タグ | Pre-filter by tag (multiselect) |
| 作成日 | Date filter: すべて / 以前 / 以降 / 範囲 |
| 参照ノート数 | Number of top-K notes to retrieve |

Filter defaults can be configured in `config.ini` via `DEFAULT_CATEGORIES`, `DEFAULT_TAGS`, `DEFAULT_DATE_MODE`, `DEFAULT_START_DATE`, `DEFAULT_END_DATE`.

### Settings page

| Feature | Description |
|---------|-------------|
| バックアップパス | Set the path to your UpNote backup directory |
| API キー | Set Anthropic and Google API keys |
| 自動前処理 | Toggle auto-preprocessing on app startup |
| 前処理を実行 | Run preprocessing with live output stream |
| データを再読み込み | Clear cache to pick up data changes without restarting |

## Embedding Model

Two embedding providers are supported, switchable via `EMBEDDING_PROVIDER` in `config.ini`:

| Provider | Model | Search |
|---|---|---|
| `local` (default) | [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) — runs locally | Hybrid: BM25 (Sudachi) + semantic (FAISS), fused via RRF |
| `google` | `gemini-embedding-001` — Google Generative AI API (3072-dim) | Semantic only (FAISS) |

For `google` provider, set `GEMINI_API_KEY` in `.env`. Each provider stores its index independently (`data/local/` and `data/google/`) and can coexist.

## Configuration Reference

All keys go in the `[settings]` section of `config.ini`. See `config.sample.ini` for a template with comments.

| Key | Default | Description |
|-----|---------|-------------|
| `UPNOTE_BACKUP_PATH` | — | Path to UpNote backup directory |
| `TOP_K_RESULTS` | `10` | Number of search results to retrieve |
| `MAX_CONTEXT_CHARS` | `50000` | Max characters passed to Claude as context |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for answer generation |
| `EMBEDDING_PROVIDER` | `local` | `local` or `google` |
| `AUTO_PREPROCESS` | `false` | Run preprocessing automatically on startup |
| `DEFAULT_DATE_MODE` | `すべて` | Default date filter mode |
| `DEFAULT_CATEGORIES` | — | Comma-separated categories to pre-select |
| `DEFAULT_TAGS` | — | Comma-separated tags to pre-select |
| `DEFAULT_START_DATE` | `2010-01-01` | Default start date for 以降/範囲 modes |
| `DEFAULT_END_DATE` | today | Default end date for 以前/範囲 modes |

## Credits

This application — including the RAG search pipeline, hybrid search implementation, and answer generation — was built with [Claude](https://claude.ai) (Anthropic).
