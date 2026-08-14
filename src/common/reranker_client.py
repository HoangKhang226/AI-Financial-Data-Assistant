"""
ViFinQA Reranker Client
──────────────────────────────────────────────────────────────
Cross-encoder reranking using FlagEmbedding (BAAI/bge-reranker-v2-m3).
Runs on CPU to preserve GPU VRAM for vLLM.
"""

from typing import List, Tuple
from src.common.logger import get_logger

logger = get_logger(__name__)


class RerankerClient:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        batch_size: int = 1,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._model = None

        logger.info(f"RerankerClient initialized: model={model_name}, device={device}")

    def load(self) -> None:
        """Load the reranker model (lazy initialization)."""
        if self._model is not None:
            return

        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                use_fp16=(self.device != "cpu"),
                device=self.device,
            )
            logger.info(f"Reranker model loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "FlagEmbedding is not installed. "
                "Run: pip install FlagEmbedding"
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The query string.
            documents: List of document texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        self.load()

        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self._model.compute_score(
            pairs, batch_size=self.batch_size, max_length=self.max_length
        )

        # Handle single score (not a list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Create (index, score) pairs and sort descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]
