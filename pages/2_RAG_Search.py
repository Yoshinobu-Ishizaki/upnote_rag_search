"""Hybrid RAG search page: BM25 + semantic search + Claude API answer generation."""
import datetime
import sys
from pathlib import Path

import polars as pl
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_search import build_bm25, load_split_data
from src.config import get_api_key, get_claude_model, get_max_context_chars, get_top_k
from src.embedding import get_embedding_model, load_index, semantic_search
from src.hybrid_search import reciprocal_rank_fusion
from src.tokenizer import create_tokenizer, tokenize_text

DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="RAG Search", layout="wide")
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
    return load_split_data(DATA_DIR)


@st.cache_data
def _load_text() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "upnote_text.csv").select(["id", "fpath", "created", "category", "tags", "contents"])


@st.cache_resource
def _bm25(_df: pl.DataFrame):
    return build_bm25(_df["tokens"])


@st.cache_resource
def _faiss_index():
    return load_index(DATA_DIR)


# ---------------------------------------------------------------------------
# Check data availability
# ---------------------------------------------------------------------------

faiss_ok = (DATA_DIR / "faiss.index").exists() and (DATA_DIR / "embeddings.npy").exists()
csv_ok = (DATA_DIR / "upnote_text_split.csv").exists() and (DATA_DIR / "upnote_text.csv").exists()

if not csv_ok:
    st.error("データファイルが見つかりません。`python preprocess.py` を実行してからアプリを再起動してください。")
    st.stop()

if not faiss_ok:
    st.warning(
        "FAISSインデックスが見つかりません。意味検索は使用できません。"
        "`python preprocess.py` を実行してインデックスを生成してください。"
    )

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

split_df = _load_split()
text_df = _load_text()
bm25 = _bm25(split_df)

if faiss_ok:
    faiss_index, id_list = _faiss_index()
    model = get_embedding_model()

top_k = get_top_k()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("フィルター")

    # Category
    all_categories = sorted([c for c in text_df["category"].drop_nulls().unique().to_list() if c])
    selected_categories = st.multiselect("カテゴリ", all_categories)

    # Tags (pipe-delimited — explode to individual tags)
    all_tags = sorted({
        tag
        for tags_str in text_df["tags"].drop_nulls().to_list()
        if tags_str
        for tag in tags_str.split("|")
        if tag
    })
    selected_tags = st.multiselect("タグ", all_tags)

    # Created date filter
    st.subheader("作成日")
    date_mode = st.selectbox("条件", ["すべて", "以前", "以降", "範囲"])
    date_before = date_after = date_from = date_to = None
    if date_mode == "以前":
        date_before = st.date_input("日付")
    elif date_mode == "以降":
        date_after = st.date_input("日付")
    elif date_mode == "範囲":
        date_range = st.date_input("期間", value=(datetime.date(2010, 1, 1), datetime.date.today()))
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            date_from, date_to = date_range

with st.form("rag_search_form"):
    question = st.text_area(
        "質問を入力してください",
        placeholder="例: Pythonでファイルを読み込む方法は？",
        height=100,
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.form_submit_button("検索・回答生成", type="primary", use_container_width=True)

if not search_btn or not question.strip():
    st.stop()

api_key = get_api_key()
if not api_key:
    st.error(
        "ANTHROPIC_API_KEY が設定されていません。**3_Settings** ページで API キーを設定してください。"
    )
    st.stop()

with st.spinner("検索中..."):
    tokenizer = _tokenizer()
    query_tokens = tokenize_text(question, tokenizer)

    # BM25 search
    scores_bm25 = bm25.get_scores(query_tokens)
    id_list_bm25 = split_df["id"].to_list()
    scored_bm25 = sorted(
        [(id_list_bm25[i], scores_bm25[i]) for i in range(len(scores_bm25)) if scores_bm25[i] > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    bm25_ids = [doc_id for doc_id, _ in scored_bm25[: top_k * 2]]

    # Semantic search (if available)
    if faiss_ok:
        query_embedding = model.encode(
            [question], normalize_embeddings=True
        )
        semantic_ids = semantic_search(faiss_index, id_list, query_embedding, top_k=top_k * 2)
    else:
        semantic_ids = []

    # Hybrid RRF fusion
    if semantic_ids:
        ranked = reciprocal_rank_fusion(bm25_ids, semantic_ids)[:top_k]
    else:
        ranked = [(doc_id, 0.0) for doc_id in bm25_ids[:top_k]]

    if not ranked:
        if not filters_active:
            st.warning("関連するノートが見つかりませんでした。キーワードを変えてお試しください。")
            st.stop()
        # else: filters active → fall through to filter-based fallback below
        ranked = []  # ensure ranked is defined for rrf_score_map later

    filters_active = bool(selected_categories or selected_tags or date_mode != "すべて")

    # Build context
    max_chars = get_max_context_chars()
    context_parts = []
    context_char_count = 0
    result_ids = [doc_id for doc_id, _ in ranked]

    text_lookup = {
        row["id"]: (row["fpath"], row["contents"] or "", row["created"], row["category"], row["tags"])
        for row in text_df.iter_rows(named=True)
    }
    print(f"[DEBUG] text_df rows={len(text_df)}, text_lookup keys={len(text_lookup)}")

    _filter_debug_done = [False]

    def _passes_filter(created, category, tags):
        def _cat_matches():
            return any(
                category == sel or category.startswith(sel + ":")
                for sel in selected_categories
            )
        debug = not _filter_debug_done[0] and selected_categories and _cat_matches()
        if selected_categories and not _cat_matches():
            return False
        if selected_tags:
            note_tags = set(tags.split("|")) if tags else set()
            if not note_tags.intersection(selected_tags):
                if debug:
                    print(f"[DEBUG] _passes_filter FAIL tags: selected_tags={selected_tags!r}, note_tags={note_tags!r}")
                    _filter_debug_done[0] = True
                return False
        if date_mode != "すべて" and created:
            try:
                note_date = datetime.date.fromisoformat(created[:10])
                if date_mode == "以前" and date_before and note_date > date_before:
                    if debug:
                        print(f"[DEBUG] _passes_filter FAIL date 以前: note_date={note_date}, date_before={date_before}")
                        _filter_debug_done[0] = True
                    return False
                if date_mode == "以降" and date_after and note_date < date_after:
                    if debug:
                        print(f"[DEBUG] _passes_filter FAIL date 以降: note_date={note_date}, date_after={date_after}")
                        _filter_debug_done[0] = True
                    return False
                if date_mode == "範囲" and date_from and date_to and not (date_from <= note_date <= date_to):
                    if debug:
                        print(f"[DEBUG] _passes_filter FAIL date 範囲: note_date={note_date}, date_from={date_from}, date_to={date_to}")
                        _filter_debug_done[0] = True
                    return False
            except ValueError:
                pass
        return True

    sample_cats = [(doc_id, repr(category)) for doc_id, (_, _, _, category, _) in list(text_lookup.items())[:5]]
    print(f"[DEBUG] selected_categories={selected_categories!r}")
    print(f"[DEBUG] sample category values in text_lookup: {sample_cats}")
    for doc_id, (_, _, created, category, tags) in list(text_lookup.items())[:200]:
        if selected_categories and any(category == sel or category.startswith(sel + ":") for sel in selected_categories):
            print(f"[DEBUG] MATCH FOUND id={doc_id} cat={repr(category)} passes={_passes_filter(created, category, tags)}")
            break
    else:
        first_cat = next(iter(text_lookup.values()), None)
        if first_cat:
            _, _, _, cat, _ = first_cat
            print(f"[DEBUG] NO MATCH in first 200: type(cat)={type(cat).__name__} cat={repr(cat)} type(sel[0])={type(selected_categories[0]).__name__ if selected_categories else 'N/A'} equal={cat == selected_categories[0] if selected_categories else 'N/A'}")

    valid_ids = {
        doc_id
        for doc_id, (fpath, contents, created, category, tags) in text_lookup.items()
        if _passes_filter(created, category, tags)
    }
    print(f"[DEBUG] valid_ids count={len(valid_ids)}, sample={list(valid_ids)[:3]}")

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
    print(f"[DEBUG] included_ids after main loop={len(included_ids)}, filters_active={filters_active}")

    # Fallback: if search results yielded no context but filters are active,
    # include ALL documents that pass the filter.
    # Claude's context window is ~200k tokens ≈ 750k chars; warn if truncated.
    CLAUDE_CONTEXT_LIMIT_CHARS = 750_000

    fallback_truncated = False
    print(f"[DEBUG] fallback check: not included_ids={not included_ids}, filters_active={filters_active}, valid_ids nonempty={bool(valid_ids)}")
    if not included_ids and filters_active and valid_ids:
        _fb_total = _fb_skipped_id = _fb_skipped_content = 0
        for row in text_df.iter_rows(named=True):
            _fb_total += 1
            doc_id = row["id"]
            if doc_id not in valid_ids:
                _fb_skipped_id += 1
                continue
            contents = row["contents"] or ""
            if not contents:
                _fb_skipped_content += 1
                continue
            fpath = row["fpath"]
            if context_char_count + len(contents) > CLAUDE_CONTEXT_LIMIT_CHARS:
                fallback_truncated = True
                break
            context_parts.append(f"--- {fpath} ---\n{contents}")
            context_char_count += len(contents)
            included_ids.append(doc_id)
        print(f"[DEBUG] fallback loop: total={_fb_total}, skipped(not in valid_ids)={_fb_skipped_id}, skipped(no content)={_fb_skipped_content}, included={len(included_ids)}")
        context = "\n\n".join(context_parts)

    if fallback_truncated:
        st.warning(
            "フィルター条件に一致するノートの合計サイズが Claude のコンテキスト上限（約 750,000 文字）を超えたため、"
            "一部のノートはコンテキストから除外されました。"
        )

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

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

st.subheader("回答")
st.markdown(message.content[0].text)

rrf_score_map = {doc_id: score for doc_id, score in ranked}

with st.expander(f"参照ノート ({len(included_ids)} 件)", expanded=False):
    for doc_id in included_ids:
        if doc_id not in text_lookup:
            continue
        fpath, contents, created, category, tags = text_lookup[doc_id]
        score = rrf_score_map.get(doc_id, 0.0)
        meta_parts = []
        if created:
            meta_parts.append(f"📅 {created}")
        if category:
            meta_parts.append(f"📁 {category}")
        if tags:
            meta_parts.append(f"🏷 {tags}")
        meta_str = " &nbsp;|&nbsp; ".join(meta_parts)
        st.markdown(f"**{fpath}** &nbsp; `RRF: {score:.4f}`")
        if meta_str:
            st.caption(meta_str, unsafe_allow_html=True)
        preview = contents[:500].replace("\n", " ") if contents else ""
        st.caption(preview + ("..." if len(contents) > 500 else ""))
        st.divider()
