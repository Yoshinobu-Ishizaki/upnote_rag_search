"""BM25 index construction and search using polars."""
from pathlib import Path

import polars as pl
from rank_bm25 import BM25Okapi


def load_split_data(data_dir: Path) -> pl.DataFrame:
    """Load tokenized note data."""
    return pl.read_csv(data_dir / "upnote_text_split.csv")


def build_bm25(tokens_series: pl.Series) -> BM25Okapi:
    """Build BM25 index from a Series of space-delimited token strings."""
    corpus = [doc.split(" ") for doc in tokens_series.to_list()]
    return BM25Okapi(corpus)


def bm25_search(
    bm25: BM25Okapi,
    df: pl.DataFrame,
    query_tokens: list[str],
    top_k: int,
) -> list[str]:
    """Return top_k doc IDs ranked by BM25 score (best first).

    Only includes documents with score > 0.
    """
    scores = bm25.get_scores(query_tokens)
    id_list = df["id"].to_list()
    scored = [
        (id_list[i], scores[i]) for i in range(len(scores)) if scores[i] > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored[:top_k]]
