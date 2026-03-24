"""FAISS-based semantic search using sentence-transformers or Google embedding API."""
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


def embed_with_google(texts: list[str], task_type: str, api_key: str) -> np.ndarray:
    """Embed texts using Google's text-multilingual-embedding-002 model.

    Args:
        texts: List of texts to embed (max 100 per call).
        task_type: 'RETRIEVAL_DOCUMENT' for indexing, 'RETRIEVAL_QUERY' for queries.
        api_key: Gemini API key.

    Returns:
        np.ndarray of shape (len(texts), 768), L2-normalized float32.
    """
    import google.genai as genai
    from google.genai import types as genai_types

    # Google API rejects empty strings — replace with a single space
    texts = [t if t.strip() else " " for t in texts]

    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=GOOGLE_EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in response.embeddings], dtype="float32")
