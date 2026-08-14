"""
ViFinQA Reranker
──────────────────────────────────────────────────────────────
BAAI/bge-reranker-v2-m3 Cross-Encoder reranking + optional
LLM Judge for evidence assessment.
"""

from typing import List, Tuple
from src.common.logger import get_logger

logger = get_logger(__name__)


def rerank_documents(
    query: str,
    doc_ids: List[str],
    doc_texts: List[str],
    reranker_client=None,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Rerank documents using cross-encoder model.

    Returns list of (doc_id, score) sorted by relevance.
    """
    if not doc_ids or not doc_texts:
        return []

    if reranker_client is None:
        # No reranker available, return original order
        return [(did, 1.0 / (i + 1)) for i, did in enumerate(doc_ids[:top_k])]

    ranked = reranker_client.rerank(query, doc_texts, top_k=top_k)
    return [(doc_ids[idx], score) for idx, score in ranked]
