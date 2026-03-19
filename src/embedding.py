"""FAISS-based semantic search using sentence-transformers."""
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st


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
) -> list[str]:
    """Return top_k doc IDs by cosine similarity (best first).

    query_embedding should be shape (1, dim) and L2-normalized.
    """
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    return [id_list[i] for i in indices[0] if 0 <= i < len(id_list)]


@st.cache_resource
def get_embedding_model():
    """Load and cache the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
