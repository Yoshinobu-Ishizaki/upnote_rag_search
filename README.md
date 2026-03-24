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

## Setup

### 1. Prepare backup file

Export your UpNote data as a native backup (`.upnx` file) from the UpNote app.

### 2. Configure settings

Launch the app and open the Settings page to set:
- Path to your `.upnx` backup file
- Anthropic API key

### 3. Preprocess data

Run the preprocessing pipeline to build the search index:

    uv run preprocess.py

This will:
1. Parse the `.upnx` file → `data/upnote_text.csv`
2. Tokenize with Sudachi → `data/upnote_text_split.csv`
3. Build FAISS semantic index → `data/faiss_index/`

To rebuild only the embeddings/FAISS index (skipping steps 1 & 2), use:

    uv run preprocess.py --embeddings-only

This is useful if you interrupted a previous run — the script saves a checkpoint after each batch and resumes automatically.

### 4. Launch the app

    uv run streamlit run app.py

## Usage

Open the **RAG Search** page, enter a query in natural language.
Optionally filter by category, tags, or date range.
The app returns a Claude-generated answer with referenced notes.

## Embedding Model

Semantic search uses the [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) model from sentence-transformers, which runs locally without any external API calls.

## Credits

This application — including the RAG search pipeline, hybrid search implementation, and answer generation — was built with [Claude](https://claude.ai) (Anthropic).
