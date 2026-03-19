"""Reciprocal Rank Fusion for combining BM25 and semantic search results."""


def reciprocal_rank_fusion(
    bm25_ranked_ids: list[str],
    semantic_ranked_ids: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1/(k + rank)  (rank is 0-based)
    k=60 is the standard default value.

    Returns:
        List of (doc_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(semantic_ranked_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
