"""
ViFinQA Embedding Client
──────────────────────────────────────────────────────────────
Wrapper for BAAI/bge-m3 using LlamaIndex HuggingFace Embedding.
Runs on CPU by default to preserve GPU VRAM for vLLM.
"""

import numpy as np
from typing import List
from src.common.logger import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Dense embedding interface using LlamaIndex + HuggingFace."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        max_length: int = 512,
        batch_size: int = 1,
        normalize: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = None

        logger.info(f"EmbeddingClient initialized: model={model_name}, device={device}")

    def load(self) -> None:
        """Load the embedding model (lazy initialization)."""
        if self._model is not None:
            return

        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            self._model = HuggingFaceEmbedding(
                model_name=self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=True,
                normalize=self.normalize,
            )
            logger.info(f"Embedding model loaded via LlamaIndex on {self.device}")
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-huggingface is not installed. "
                "Run: pip install llama-index-embeddings-huggingface"
            )

    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into dense vectors.

        Args:
            texts: List of text strings to embed.
            show_progress: Show progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), dimension).
        """
        self.load()

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embs = [self._model.get_text_embedding(t) for t in batch]
            embeddings.extend(batch_embs)
            if show_progress and (i + self.batch_size) % 50 == 0:
                logger.info(f"Embedding progress: {min(i + self.batch_size, len(texts))}/{len(texts)}")

        return np.array(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string.

        Returns:
            1D np.ndarray of shape (dimension,).
        """
        self.load()
        return np.array(self._model.get_text_embedding(text))

    def get_llama_index_embedding(self):
        """Return the underlying LlamaIndex embedding object for direct use."""
        self.load()
        return self._model
