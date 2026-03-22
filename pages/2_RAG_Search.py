"""Hybrid RAG search page: BM25 + semantic search + Claude API answer generation."""
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
        st.warning("関連するノートが見つかりませんでした。キーワードを変えてお試しください。")
        st.stop()

    # Build context
    max_chars = get_max_context_chars()
    context_parts = []
    context_char_count = 0
    result_ids = [doc_id for doc_id, _ in ranked]

    text_lookup = {
        row["id"]: (row["fpath"], row["contents"] or "", row["created"], row["category"], row["tags"])
        for row in text_df.iter_rows(named=True)
    }

    included_ids = []
    for doc_id in result_ids:
        if doc_id not in text_lookup:
            continue
        fpath, contents, *_ = text_lookup[doc_id]
        if context_char_count + len(contents) > max_chars:
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
