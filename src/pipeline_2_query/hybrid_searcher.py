"""
ViFinQA Hybrid Searcher
──────────────────────────────────────────────────────────────
BM25 Sparse + BGE-M3 Dense + RRF Fusion search.
Uses LlamaIndex embedding client for dense search.
"""

import json
import pickle
import numpy as np
from typing import List, Tuple, Optional
from src.common.logger import get_logger

logger = get_logger(__name__)


class HybridSearcher:
    """Performs hybrid dense + sparse search with RRF fusion."""

    def __init__(
        self,
        dense_dir: str = "data/index/dense_vectors",
        bm25_dir: str = "data/index/bm25_index",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.dense_dir = dense_dir
        self.bm25_dir = bm25_dir
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self._dense_vectors = None
        self._dense_doc_ids = None
        self._bm25 = None
        self._bm25_doc_ids = None

    def load(self) -> None:
        """Load both indices."""
        self._load_dense()
        self._load_sparse()

    def _load_dense(self) -> None:
        import os
        vec_path = os.path.join(self.dense_dir, "vectors.npy")
        ids_path = os.path.join(self.dense_dir, "doc_ids.json")
        if os.path.exists(vec_path):
            self._dense_vectors = np.load(vec_path)
            with open(ids_path, encoding="utf-8") as f:
                self._dense_doc_ids = json.load(f)
            logger.info(f"Loaded {len(self._dense_doc_ids)} dense vectors")

    def _load_sparse(self) -> None:
        import os
        bm25_path = os.path.join(self.bm25_dir, "bm25.pkl")
        ids_path = os.path.join(self.bm25_dir, "doc_ids.json")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
            with open(ids_path, encoding="utf-8") as f:
                self._bm25_doc_ids = json.load(f)
            logger.info(f"Loaded BM25 index with {len(self._bm25_doc_ids)} docs")

    def search(
        self,
        query: str,
        top_k: int = 20,
        scope_doc_ids: Optional[List[str]] = None,
        embedding_client=None,
    ) -> List[Tuple[str, float]]:
        """Hybrid search returning (doc_id, score) pairs."""
        dense_results = self._dense_search(query, top_k * 2, embedding_client)
        sparse_results = self._sparse_search(query, top_k * 2)

        # RRF Fusion
        fused = self._rrf_fusion(dense_results, sparse_results)

        # Scope filtering
        if scope_doc_ids:
            scope_set = set(scope_doc_ids)
            fused = [(did, s) for did, s in fused if did in scope_set]

        return fused[:top_k]

    def _dense_search(
        self, query: str, top_k: int, embedding_client=None,
    ) -> List[Tuple[str, float]]:
        if self._dense_vectors is None or embedding_client is None:
            return []

        q_vec = embedding_client.encode_single(query)
        scores = np.dot(self._dense_vectors, q_vec)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self._dense_doc_ids[i], float(scores[i])) for i in top_idx
        ]

    def _sparse_search(
        self, query: str, top_k: int,
    ) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self._bm25_doc_ids[i], float(scores[i]))
            for i in top_idx if scores[i] > 0
        ]

    def _rrf_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion."""
        scores = {}
        for rank, (doc_id, _) in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + (
                self.dense_weight / (self.rrf_k + rank + 1)
            )
        for rank, (doc_id, _) in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0) + (
                self.sparse_weight / (self.rrf_k + rank + 1)
            )
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
