"""
ViFinQA Reranker Client
──────────────────────────────────────────────────────────────
Cross-encoder reranking using pure transformers (BAAI/bge-reranker-v2-m3).
Bypasses FlagEmbedding and sentence-transformers dependency issues.
Runs on CPU to preserve GPU VRAM for vLLM.
"""

from typing import List, Tuple
from src.common.logger import get_logger

logger = get_logger(__name__)


class RerankerClient:
    """Cross-encoder reranker using pure HuggingFace Transformers."""

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
        self._tokenizer = None

        logger.info(f"RerankerClient initialized: model={model_name}, device={device} (Pure Transformers)")

    def load(self) -> None:
        """Load the reranker model (lazy initialization)."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._torch = torch
            
            logger.info(f"Reranker model loaded on {self.device} via AutoModelForSequenceClassification")
        except Exception as e:
            raise ImportError(
                f"Failed to load Reranker via transformers. Original error: {e}. "
                "Ensure transformers is installed correctly."
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query."""
        self.load()

        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        
        # Process in single batch (or smaller batches if memory is an issue)
        with self._torch.no_grad():
            inputs = self._tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.max_length
            ).to(self.device)
            
            outputs = self._model(**inputs, return_dict=True)
            scores = outputs.logits.view(-1,).float().cpu().numpy()

        # Handle single score (not a list/array)
        if scores.ndim == 0:
            scores = [scores.item()]
        else:
            scores = scores.tolist()

        # Create (index, score) pairs and sort descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]
