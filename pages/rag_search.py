"""Hybrid RAG search page: BM25 + semantic search + Claude API answer generation."""
import datetime
import sys
from pathlib import Path

import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from st_aggrid import AgGrid, GridOptionsBuilder

from src.bm25_search import build_bm25, load_split_data
from src.config import (
    get_api_key,
    get_claude_model,
    get_default_categories,
    get_default_date_mode,
    get_default_end_date,
    get_default_start_date,
    get_default_tags,
    get_embedding_provider,
    get_gemini_api_key,
    get_max_context_chars,
    get_top_k,
)
from src.embedding import embed_with_google, get_embedding_model, load_index, semantic_search
from src.hybrid_search import reciprocal_rank_fusion
from src.date_extractor import extract_date_range
from src.tokenizer import create_tokenizer, tokenize_text

DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DATA_DIR = DATA_DIR / "local"
GOOGLE_DATA_DIR = DATA_DIR / "google"
PROVIDER = get_embedding_provider()

if PROVIDER == "google":
    st.title("RAG Search")
    st.caption(f"意味検索（Google Embedding）+ Claude AI による回答生成 | モデル: `{get_claude_model()}`")
else:
    st.title("RAG Search")
    st.caption(f"ハイブリッド検索（BM25 + 意味検索）+ Claude AI による回答生成 | モデル: `{get_claude_model()}`")

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def _tokenizer():
    return create_tokenizer()


@st.cache_data
def _load_split() -> pl.DataFrame:
    return load_split_data(LOCAL_DATA_DIR)


@st.cache_data
def _load_text() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "upnote_text.csv").select(["id", "fpath", "created", "category", "tags", "contents"])


@st.cache_resource
def _bm25(_df: pl.DataFrame):
    return build_bm25(_df["tokens"])


@st.cache_resource
def _faiss_index_local():
    return load_index(LOCAL_DATA_DIR)


@st.cache_resource
def _faiss_index_google():
    return load_index(GOOGLE_DATA_DIR)


# ---------------------------------------------------------------------------
# Check data availability
# ---------------------------------------------------------------------------

if PROVIDER == "google":
    faiss_ok = (GOOGLE_DATA_DIR / "faiss.index").exists()
    csv_ok = (DATA_DIR / "upnote_text.csv").exists()
else:
    faiss_ok = (LOCAL_DATA_DIR / "faiss.index").exists() and (LOCAL_DATA_DIR / "embeddings.npy").exists()
    csv_ok = (LOCAL_DATA_DIR / "upnote_text_split.csv").exists() and (DATA_DIR / "upnote_text.csv").exists()

if not csv_ok:
    st.error("データファイルが見つかりません。`python preprocess.py` を実行してからアプリを再起動してください。")
    st.stop()

if not faiss_ok:
    st.warning(
        "FAISSインデックスが見つかりません。"
        "`python preprocess.py` を実行してインデックスを生成してください。"
    )

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

text_df = _load_text()

if PROVIDER == "google":
    if faiss_ok:
        faiss_index, id_list = _faiss_index_google()
else:
    split_df = _load_split()
    bm25 = _bm25(split_df)
    if faiss_ok:
        faiss_index, id_list = _faiss_index_local()
        model = get_embedding_model()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("フィルター")

    # Category
    all_categories = sorted([c for c in text_df["category"].drop_nulls().unique().to_list() if c])
    _default_cats = [c for c in get_default_categories() if c in all_categories]
    selected_categories = st.multiselect("カテゴリ", all_categories, default=_default_cats)

    # Tags (pipe-delimited — explode to individual tags)
    all_tags = sorted({
        tag
        for tags_str in text_df["tags"].drop_nulls().to_list()
        if tags_str
        for tag in tags_str.split("|")
        if tag
    })
    _default_tags = [t for t in get_default_tags() if t in all_tags]
    selected_tags = st.multiselect("タグ", all_tags, default=_default_tags)

    # Created date filter
    st.subheader("作成日")
    _date_modes = ["すべて", "以前", "以降", "範囲"]
    _default_date_mode = get_default_date_mode()
    date_mode = st.selectbox("条件", _date_modes, index=_date_modes.index(_default_date_mode))
    date_before = date_after = date_from = date_to = None
    _min_date = datetime.date(1998, 1, 1)
    _max_date = datetime.date.today()
    if date_mode == "以前":
        date_before = st.date_input("日付", value=get_default_end_date(), min_value=_min_date, max_value=_max_date)
    elif date_mode == "以降":
        date_after = st.date_input("日付", value=get_default_start_date(), min_value=_min_date, max_value=_max_date)
    elif date_mode == "範囲":
        date_range = st.date_input("期間", value=(get_default_start_date(), get_default_end_date()), min_value=_min_date, max_value=_max_date)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            date_from, date_to = date_range

    st.subheader("検索設定")
    top_k = st.number_input("参照ノート数", min_value=1, max_value=100, value=get_top_k(), step=1)

with st.form("rag_search_form"):
    question = st.text_area(
        "質問を入力してください",
        placeholder="例: Pythonでファイルを読み込む方法は？",
        height=100,
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.form_submit_button("検索・回答生成", type="primary", use_container_width=True)

new_search = search_btn and bool(question.strip())
has_results = "rag_results" in st.session_state

if not new_search and not has_results:
    st.stop()

if new_search:
    if "rag_results" in st.session_state:
        del st.session_state["rag_results"]

    api_key = get_api_key()
    if not api_key:
        st.error(
            "ANTHROPIC_API_KEY が設定されていません。**Settings** ページで API キーを設定してください。"
        )
        st.stop()

    # Auto-detect date range from question
    _max_note_date_str = text_df["created"].drop_nulls().map_elements(lambda s: s[:10], return_dtype=pl.Utf8).max()
    _max_note_date = datetime.date.fromisoformat(_max_note_date_str) if _max_note_date_str else None
    _auto_date = extract_date_range(question, api_key, max_note_date=_max_note_date)
    if _auto_date:
        date_mode = "範囲"
        date_from = _auto_date["date_from"]
        date_to = _auto_date["date_to"]
        st.session_state["auto_date_notice"] = _auto_date
    else:
        st.session_state.pop("auto_date_notice", None)

    with st.spinner("検索中..."):
        if PROVIDER == "google":
            # Google embedding: semantic search only
            if faiss_ok:
                gemini_api_key = get_gemini_api_key()
                try:
                    query_embedding = embed_with_google([question], "RETRIEVAL_QUERY", gemini_api_key)
                    ranked = semantic_search(faiss_index, id_list, query_embedding, top_k=top_k)
                except RuntimeError as e:
                    st.error(str(e))
                    st.info("Set `EMBEDDING_PROVIDER=local` in config.ini and restart the app to use local embeddings.")
                    st.stop()
            else:
                ranked = []
        else:
            # Local: hybrid BM25 + semantic search
            tokenizer = _tokenizer()
            query_tokens = tokenize_text(question, tokenizer)

            scores_bm25 = bm25.get_scores(query_tokens)
            id_list_bm25 = split_df["id"].to_list()
            scored_bm25 = sorted(
                [(id_list_bm25[i], scores_bm25[i]) for i in range(len(scores_bm25)) if scores_bm25[i] > 0],
                key=lambda x: x[1],
                reverse=True,
            )
            bm25_ids = [doc_id for doc_id, _ in scored_bm25[: top_k * 2]]

            if faiss_ok:
                query_embedding = model.encode([question], normalize_embeddings=True)
                semantic_results = semantic_search(faiss_index, id_list, query_embedding, top_k=top_k * 2)
                semantic_ids = [doc_id for doc_id, _ in semantic_results]
            else:
                semantic_ids = []

            if semantic_ids:
                ranked = reciprocal_rank_fusion(bm25_ids, semantic_ids)[:top_k]
            else:
                ranked = [(doc_id, 0.0) for doc_id in bm25_ids[:top_k]]

        filters_active = bool(selected_categories or selected_tags or date_mode != "すべて")

        if not ranked:
            if not filters_active:
                st.warning("関連するノートが見つかりませんでした。キーワードを変えてお試しください。")
                st.stop()
            ranked = []

        # Build context
        max_chars = get_max_context_chars()
        context_parts = []
        context_char_count = 0
        result_ids = [doc_id for doc_id, _ in ranked]

        text_lookup = {
            row["id"]: (row["fpath"], row["contents"] or "", row["created"], row["category"], row["tags"])
            for row in text_df.iter_rows(named=True)
        }

        def _passes_filter(created, category, tags):
            def _cat_matches():
                return any(
                    category == sel or category.startswith(sel + ":")
                    for sel in selected_categories
                )
            if selected_categories and not _cat_matches():
                return False
            if selected_tags:
                note_tags = set(tags.split("|")) if tags else set()
                if not note_tags.intersection(selected_tags):
                    return False
            if date_mode != "すべて" and created:
                try:
                    note_date = datetime.date.fromisoformat(created[:10])
                    if date_mode == "以前" and date_before and note_date > date_before:
                        return False
                    if date_mode == "以降" and date_after and note_date < date_after:
                        return False
                    if date_mode == "範囲" and date_from and date_to and not (date_from <= note_date <= date_to):
                        return False
                except ValueError:
                    pass
            return True

        valid_ids = {
            doc_id
            for doc_id, (fpath, contents, created, category, tags) in text_lookup.items()
            if _passes_filter(created, category, tags)
        }

        included_ids = []
        for doc_id in result_ids:
            if doc_id not in text_lookup or doc_id not in valid_ids:
                continue
            fpath, contents, *_ = text_lookup[doc_id]
            if not contents:
                continue
            if context_char_count + len(contents) > max_chars:
                break
            context_parts.append(f"--- {fpath} ---\n{contents}")
            context_char_count += len(contents)
            included_ids.append(doc_id)

        context = "\n\n".join(context_parts)

        # Fallback: if search results yielded no context but filters are active,
        # include ALL documents that pass the filter.
        # Claude's context window is ~200k tokens ≈ 750k chars; warn if truncated.
        CLAUDE_CONTEXT_LIMIT_CHARS = 750_000

        fallback_truncated = False
        if not included_ids and filters_active and valid_ids:
            for row in text_df.iter_rows(named=True):
                doc_id = row["id"]
                if doc_id not in valid_ids:
                    continue
                contents = row["contents"] or ""
                if not contents:
                    continue
                fpath = row["fpath"]
                if context_char_count + len(contents) > CLAUDE_CONTEXT_LIMIT_CHARS:
                    fallback_truncated = True
                    break
                context_parts.append(f"--- {fpath} ---\n{contents}")
                context_char_count += len(contents)
                included_ids.append(doc_id)
            context = "\n\n".join(context_parts)

    with st.spinner("Claude に問い合わせ中..."):
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=get_claude_model(),
            max_tokens=2048,
            system=(
                "提供されたノートのコンテキストのみを使って質問に答えてください。"
                "コンテキストに答えが見つからない場合はその旨を伝えてください。"
                "質問と同じ言語で回答してください。"
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"コンテキスト:\n\n{context}\n\n質問: {question}",
                }
            ],
        )

    score_map = {doc_id: score for doc_id, score in ranked}
    score_col = "スコア"

    rows = []
    for doc_id in included_ids:
        if doc_id not in text_lookup:
            continue
        fpath, contents, created, category, tags = text_lookup[doc_id]
        score = score_map.get(doc_id, 0.0)
        preview = (contents[:300] + "...") if contents and len(contents) > 300 else (contents or "")
        rows.append({
            "ノート": fpath,
            "カテゴリ": category or "",
            "作成日": created[:10] if created else "",
            "タグ": tags or "",
            "内容プレビュー": preview,
            score_col: round(score, 4),
        })

    result_df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "ノート": pl.Utf8, "カテゴリ": pl.Utf8, "作成日": pl.Utf8,
        "タグ": pl.Utf8, "内容プレビュー": pl.Utf8, score_col: pl.Float64,
    })

    st.session_state["rag_results"] = {
        "answer": message.content[0].text,
        "result_df": result_df,
        "included_ids": included_ids,
        "fallback_truncated": fallback_truncated,
        "score_col": score_col,
    }

    if fallback_truncated:
        st.warning(
            "フィルター条件に一致するノートの合計サイズが Claude のコンテキスト上限（約 750,000 文字）を超えたため、"
            "一部のノートはコンテキストから除外されました。"
        )

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

res = st.session_state["rag_results"]

if st.session_state.get("auto_date_notice"):
    _n = st.session_state["auto_date_notice"]
    st.info(
        f"日付範囲を自動検出しました: **{_n['date_from']}** ～ **{_n['date_to']}**  "
        "（サイドバーで手動変更可能）"
    )

st.subheader("回答")
st.markdown(res["answer"])

result_df = res["result_df"]
included_ids = res["included_ids"]
score_col = res.get("score_col", "スコア")

st.subheader(f"参照ノート ({len(included_ids)} 件)")

gb = GridOptionsBuilder.from_dataframe(result_df.to_pandas())
gb.configure_default_column(
    filter=True,
    sortable=True,
    resizable=True,
    wrapText=True,
    autoHeight=True,
)
gb.configure_grid_options(enableCellTextSelection=True)
gb.configure_column("内容プレビュー", width=500)
gb.configure_column("ノート", width=180)
gb.configure_column("カテゴリ", width=100)
gb.configure_column("作成日", width=100)
gb.configure_column("タグ", width=100)
gb.configure_column(score_col, width=90, type=["numericColumn"], valueFormatter="x.toFixed(4)")

AgGrid(
    result_df.to_pandas(),
    gridOptions=gb.build(),
    width="stretch",
    height=400,
    fit_columns_on_grid_load=True,
)
