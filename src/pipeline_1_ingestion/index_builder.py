"""
ViFinQA Index Builder
──────────────────────────────────────────────────────────────
Builds BGE-M3 dense embedding vectors and BM25Okapi sparse index
from table summaries for hybrid retrieval.
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Optional


class IndexBuilder:
    """Builds and persists dense + sparse indices."""

    def __init__(
        self,
        dense_dir: str = "data/index/dense_vectors",
        bm25_dir: str = "data/index/bm25_index",
    ):
        self.dense_dir = dense_dir
        self.bm25_dir = bm25_dir

    def build_dense_index(
        self,
        summaries: List[dict],
        embedding_client=None,
    ) -> None:
        """Build dense vector index from summaries using BGE-M3."""
        os.makedirs(self.dense_dir, exist_ok=True)

        if embedding_client is None:
            from src.common.embedding_client import EmbeddingClient
            embedding_client = EmbeddingClient()

        texts = [s["summary"] for s in summaries]
        doc_ids = [s["doc_id"] for s in summaries]

        vectors = embedding_client.encode(texts, show_progress=True)

        np.save(os.path.join(self.dense_dir, "vectors.npy"), vectors)
        with open(os.path.join(self.dense_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, ensure_ascii=False)

    def build_sparse_index(self, summaries: List[dict]) -> None:
        """Build BM25Okapi sparse index from summaries."""
        os.makedirs(self.bm25_dir, exist_ok=True)

        from rank_bm25 import BM25Okapi

        corpus = []
        doc_ids = []
        for s in summaries:
            tokens = s["summary"].lower().split()
            tokens.extend(s.get("key_terms", []))
            corpus.append(tokens)
            doc_ids.append(s["doc_id"])

        bm25 = BM25Okapi(corpus)

        with open(os.path.join(self.bm25_dir, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25, f)
        with open(os.path.join(self.bm25_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, ensure_ascii=False)
        with open(os.path.join(self.bm25_dir, "corpus.json"), "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)

    def build_all(self, summaries_path: str, embedding_client=None) -> None:
        """Build both indices from a summaries.jsonl file."""
        summaries = []
        with open(summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    summaries.append(json.loads(line))

        self.build_sparse_index(summaries)
        if embedding_client:
            self.build_dense_index(summaries, embedding_client)
