"""FAISS-based semantic search using sentence-transformers or Google embedding API."""
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"
GOOGLE_EMBEDDING_DIM = 3072


def load_index(data_dir: Path) -> tuple:
    """Load FAISS index and corresponding doc ID list.

    Returns:
        (faiss.Index, list[str]) — index and ordered list of doc IDs
    """
    import faiss  # imported here to keep startup fast if faiss unavailable

    index = faiss.read_index(str(data_dir / "faiss.index"))
    df = pl.read_csv(data_dir / "upnote_text_split.csv")
    id_list = df["id"].to_list()
    return index, id_list


def semantic_search(
    index,
    id_list: list[str],
    query_embedding: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """Return top_k (doc_id, score) tuples by cosine similarity (best first).

    query_embedding should be shape (1, dim) and L2-normalized.
    """
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    return [
        (id_list[i], float(distances[0][rank]))
        for rank, i in enumerate(indices[0])
        if 0 <= i < len(id_list)
    ]


@st.cache_resource
def get_embedding_model():
    """Load and cache the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


_RATE_LIMIT_HELP = (
    "Google Embedding API free-tier limit (100 req/min) exceeded after all retries.\n"
    "Options:\n"
    "  1. Switch to local embeddings: set EMBEDDING_PROVIDER=local in config.ini\n"
    "  2. Upgrade to a paid Google AI plan\n"
    "     https://ai.google.dev/gemini-api/docs/rate-limits"
)
_MAX_RETRIES = 5
_DEFAULT_RETRY_DELAY = 60  # seconds


def _is_rate_limit_error(e: Exception) -> bool:
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


def _parse_retry_delay(e: Exception) -> float:
    """Extract retry delay in seconds from a 429 error message."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(e))
    if m:
        return float(m.group(1))
    # Also try plain "retry in Xs" phrasing
    m2 = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e))
    if m2:
        return float(m2.group(1))
    return _DEFAULT_RETRY_DELAY


def embed_with_google(texts: list[str], task_type: str, api_key: str) -> np.ndarray:
    """Embed texts using Google's Gemini embedding model with automatic retry on rate limit.

    Args:
        texts: List of texts to embed (max 100 per call).
        task_type: 'RETRIEVAL_DOCUMENT' for indexing, 'RETRIEVAL_QUERY' for queries.
        api_key: Gemini API key.

    Returns:
        np.ndarray of shape (len(texts), GOOGLE_EMBEDDING_DIM), L2-normalized float32.

    Raises:
        RuntimeError: When rate limit is exceeded after all retries, with guidance for the user.
    """
    import google.genai as genai
    from google.genai import types as genai_types

    # Google API rejects empty strings — replace with a single space
    texts = [t if t.strip() else " " for t in texts]

    client = genai.Client(api_key=api_key)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=GOOGLE_EMBEDDING_MODEL,
                contents=texts,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )
            return np.array([e.values for e in response.embeddings], dtype="float32")
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < _MAX_RETRIES - 1:
                delay = _parse_retry_delay(e)
                print(
                    f"  Rate limit hit. Waiting {delay:.0f}s before retry "
                    f"({attempt + 1}/{_MAX_RETRIES - 1})..."
                )
                time.sleep(delay + 2)
            else:
                if _is_rate_limit_error(e):
                    raise RuntimeError(_RATE_LIMIT_HELP) from e
                raise
